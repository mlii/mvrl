import json
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

import config
from data.dataset import get_dm
from env import make_env
from ops import build_vae, build_vaes, build_rnn, build_rnns, get_vmmd_losses, \
  get_rmmd_losses, get_transform_loss, get_predicted_transform_loss, \
  get_vae_rec_ops, get_vae_pred_ops
from models.vrnn import VRNN
from utils import saveToFlat, check_dir, get_output_log
from models.vae import ConvVAE
from wrappers import WrapperFactory

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)


def learn(sess, n_tasks, z_size, data_dir, num_steps, max_seq_len,
          batch_size_per_task=16, rnn_size=256,
          grad_clip=1.0, v_lr=0.0001, vr_lr=0.0001,
          min_v_lr=0.00001, v_decay=0.999, kl_tolerance=0.5,
          lr=0.001, min_lr=0.00001, decay=0.999,
          view="transposed",
          model_dir="tf_rnn", layer_norm=False,
          rnn_mmd=False, no_cor=False,
          w_mmd=1.0,
          alpha=1.0, beta=0.1,
          recurrent_dp=1.0,
          input_dp=1.0,
          output_dp=1.0):
  batch_size = batch_size_per_task * n_tasks

  wrapper = WrapperFactory.get_wrapper(view)
  if wrapper is None:
    raise Exception("Such view is not available")

  print("Batch size for each taks is", batch_size_per_task)
  print("The total batch size is", batch_size)

  check_dir(model_dir)
  lf = open(model_dir + '/log_%s' % datetime.now().isoformat(), "w")
  # define env
  na = make_env(config.env_name).action_space.n
  input_size = z_size + na
  output_size = z_size
  print("the environment", config.env_name, "has %i actions" % na)

  seq_len = max_seq_len

  fns = os.listdir(data_dir)
  fns = [fn for fn in fns if '.npz' in fn]
  random.shuffle(fns)
  dm = get_dm(wrapper, seq_len, na, data_dir, fns, not no_cor)
  tf_vrct_lr = tf.placeholder(tf.float32,
                              shape=[])  # learn from reconstruction.
  vaes, vcomps = build_vaes(n_tasks, na, z_size, seq_len, tf_vrct_lr,
                            kl_tolerance)
  vae_losses = [vcomp.loss for vcomp in vcomps]
  transform_loss = get_transform_loss(vcomps[0], vaes[1], wrapper)

  old_vae0 = ConvVAE(name="old_vae0", z_size=z_size)
  old_vcomp0 = build_vae("old_vae0", old_vae0, na, z_size, seq_len,
                         tf_vrct_lr, kl_tolerance)
  assign_old_eq_new = tf.group([tf.assign(oldv, newv)
                                for (oldv, newv) in
                                zip(old_vcomp0.var_list, vcomps[0].var_list)])

  vmmd_losses = get_vmmd_losses(n_tasks, old_vcomp0, vcomps, alpha, beta)
  vrec_ops = get_vae_rec_ops(n_tasks, vcomps, vmmd_losses, w_mmd)
  vrec_all_op = tf.group(vrec_ops)

  # Meta RNN.
  rnn = VRNN("rnn", max_seq_len, input_size, output_size, batch_size_per_task,
             rnn_size, layer_norm, recurrent_dp, input_dp, output_dp)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  tf_rpred_lr = tf.placeholder(tf.float32, shape=[])
  rcomp0 = build_rnn("rnn", rnn, na, z_size, batch_size_per_task, seq_len)

  print("The basic rnn has been built")

  rcomps = build_rnns(n_tasks, rnn, vaes, vcomps, kl_tolerance)
  rnn_losses = [rcomp.loss for rcomp in rcomps]

  if rnn_mmd:
    rmmd_losses = get_rmmd_losses(n_tasks, old_vcomp0, vcomps, alpha, beta)
    for i in range(n_tasks):
      rnn_losses[i] += 0.1 * rmmd_losses[i]

  ptransform_loss = get_predicted_transform_loss(vcomps[0], rcomps[0],
                                                 vaes[1],
                                                 wrapper, batch_size_per_task,
                                                 seq_len)
  print("RNN has been connected to each VAE")

  rnn_total_loss = tf.reduce_mean(rnn_losses)
  rpred_opt = tf.train.AdamOptimizer(tf_rpred_lr, name="rpred_opt")
  gvs = rpred_opt.compute_gradients(rnn_total_loss, rcomp0.var_list)
  clip_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for
              grad, var in gvs if grad is not None]
  rpred_op = rpred_opt.apply_gradients(clip_gvs, global_step=global_step,
                                       name='rpred_op')

  # VAE in prediction phase
  vpred_ops, tf_vpred_lrs = get_vae_pred_ops(n_tasks, vcomps, rnn_losses)
  vpred_all_op = tf.group(vpred_ops)

  rpred_lr = lr
  vrct_lr = v_lr
  vpred_lr = vr_lr
  sess.run(tf.global_variables_initializer())

  for i in range(num_steps):

    step = sess.run(global_step)
    rpred_lr = (rpred_lr - min_lr) * decay + min_lr
    vrct_lr = (vrct_lr - min_v_lr) * v_decay + min_v_lr
    vpred_lr = (vpred_lr - min_v_lr) * v_decay + min_v_lr

    ratio = 1.0

    data_buffer = []

    for it in range(config.psteps_per_it):
      raw_obs_list, raw_a_list = dm.random_batch(batch_size_per_task)
      data_buffer.append((raw_obs_list, raw_a_list))

      feed = {tf_rpred_lr: rpred_lr, tf_vrct_lr: vrct_lr,
              tf_vpred_lrs[0]: vpred_lr,
              tf_vpred_lrs[1]: vpred_lr * ratio}
      feed[old_vcomp0.x] = raw_obs_list[0]
      for j in range(n_tasks):
        vcomp = vcomps[j]
        feed[vcomp.x] = raw_obs_list[j]
        feed[vcomp.a] = raw_a_list[j][:, :-1, :]

      (rnn_cost, rnn_cost2, vae_cost, vae_cost2,
       transform_cost, ptransform_cost, _, _) = sess.run(
        [rnn_losses[0], rnn_losses[1],
         vae_losses[0], vae_losses[1],
         transform_loss, ptransform_loss,
         rpred_op, vpred_all_op], feed)
      ratio = rnn_cost2 / rnn_cost

    if i % config.log_interval == 0:
      output_log = get_output_log(step, rpred_lr, [vae_cost], [rnn_cost], [transform_cost], [ptransform_cost])
      lf.write(output_log)

    data_order = np.arange(len(data_buffer))
    nd = len(data_order)
    np.random.shuffle(data_order)

    for it in range(config.rsteps_per_it):
      if (it + 1) % nd == 0:
        np.random.shuffle(data_order)
      rid = data_order[it % nd]

      raw_obs_list, raw_a_list = data_buffer[rid]
      # raw_obs_list, raw_a_list = dm.random_batch(batch_size_per_task)

      feed = {tf_rpred_lr: rpred_lr, tf_vrct_lr: vrct_lr}
      feed[old_vcomp0.x] = raw_obs_list[0]
      for j in range(n_tasks):
        vcomp = vcomps[j]
        feed[vcomp.x] = raw_obs_list[j]
        feed[vcomp.a] = raw_a_list[j][:, :-1, :]

      (rnn_cost, rnn_cost2, vae_cost, vae_cost2, transform_cost,
       ptransform_cost, _) = sess.run([
        rnn_losses[0], rnn_losses[1],
        vae_losses[0], vae_losses[1],
        transform_loss, ptransform_loss,
        vrec_all_op], feed)

    if i % config.log_interval == 0:
      output_log = get_output_log(step, rpred_lr, [vae_cost], [rnn_cost], [transform_cost], [ptransform_cost])
      lf.write(output_log)

    lf.flush()

    if (i + 1) % config.target_update_interval == 0:
      sess.run(assign_old_eq_new)

    if i % config.model_save_interval == 0:
      tmp_dir = model_dir + '/it_%i' % i
      check_dir(tmp_dir)
      saveToFlat(rcomp0.var_list, tmp_dir + '/rnn.p')
      for j in range(n_tasks):
        vcomp = vcomps[j]
        saveToFlat(vcomp.var_list, tmp_dir + '/vae%i.p' % j)

  saveToFlat(rcomp0.var_list, model_dir + '/final_rnn.p')
  for i in range(n_tasks):
    vcomp = vcomps[i]
    saveToFlat(vcomp.var_list, model_dir + '/final_vae%i.p' % i)


def main():
  import argparse
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--z-size', type=int, default=32, help="z of VAE")
  parser.add_argument('--data-dir', default="record",
                      help="the data directory")
  parser.add_argument('--max-seq-len', type=int, default=25,
                      help="the maximum steps of dynamics to catch")
  parser.add_argument('--num-steps', type=int, default=4000,
                      help="number of training iterations")
  parser.add_argument('--batch-size-per-task', type=int, default=16,
                      help="batch size for each task")
  parser.add_argument('--rnn-size', type=int, default=32,
                      help="rnn hidden state size")
  parser.add_argument('--grad-clip', type=float, default=1.0,
                      help="grad clip range")
  parser.add_argument('--lr', type=float, default=0.001,
                      help="learning rate")
  parser.add_argument('--min-lr', type=float, default=0.00001,
                      help="minimum of learning rate")
  parser.add_argument('--decay', type=float, default=0.99999,
                      help="decay of learning rate")
  parser.add_argument('--view', default="transposed",
                      help="type of view: transposed, mirror, h-swapped, inverse.")
  parser.add_argument('--n-tasks', type=int, default=2,
                      help="the number of tasks")
  parser.add_argument('--v-lr', type=float, default=0.0001,
                      help="the learning rate of vae")
  parser.add_argument('--vr-lr', type=float, default=0.0001,
                      help="the learning rate of vae to reduce the rnn loss")
  parser.add_argument('--min-v-lr', type=float, default=0.00001,
                      help="the minimum of vae learning rate")
  parser.add_argument('--v-decay', type=float, default=1.0,
                      help="the decay of vae learning rare")
  parser.add_argument('--kl-tolerance', type=float, default=0.5,
                      help="kl tolerance")
  parser.add_argument('--w-mmd', type=float, default=0.5,
                      help="the weight of MMD loss")
  parser.add_argument('--alpha', type=float, default=1.0,
                      help="the weight of MMD mean loss")
  parser.add_argument('--beta', type=float, default=0.1,
                      help="the weight MMD logstd loss")
  parser.add_argument('--model-dir', default="tf_rnn",
                      help="the directory to store rnn model")
  parser.add_argument('--no-cor', action="store_true", default=False,
                      help="Not use the corresponding input")
  parser.add_argument('--layer-norm', action="store_true", default=False,
                      help="layer norm in RNN")
  parser.add_argument('--rnn-mmd', action="store_true", default=False,
                      help="apply mmd loss in rnn")
  parser.add_argument('--recurrent-dp', type=float, default=1.0,
                      help="dropout ratio in recurrent")
  parser.add_argument('--input-dp', type=float, default=1.0,
                      help="dropout ratio in input")
  parser.add_argument('--output-dp', type=float, default=1.0,
                      help="dropout ratio in output")
  parser.add_argument('--gpu', default="0", help="which gpu to use")

  args = vars(parser.parse_args())

  check_dir(args["model_dir"])
  os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
  del (args["gpu"])

  seed = 1234567
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  args2login = {"seed": seed}
  args2login.update(args)

  with open(args["model_dir"] + '/args.json', "w") as f:
    json.dump(args2login, f, indent=2, sort_keys=True)

  tf_config = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False)
  tf_config.gpu_options.allow_growth = True
  with tf.Session(config=tf_config) as sess:
    learn(sess, **args)


if __name__ == '__main__':
  main()
