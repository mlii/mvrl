from collections import namedtuple

import tensorflow as tf

from utils import get_lr_lossfunc, get_kl2normal_lossfunc
from models.vae import ConvVAE

VAE_COMP = namedtuple('VAE_COMP',
                      ['a', 'x', 'y', 'z', 'mean', 'logstd', 'r_loss',
                       'kl_loss', 'loss', 'var_list', 'fc_var_list',
                       'train_opt'])
RNN_COMP_WITH_OPT = namedtuple('RNN_COMP',
                               ['z_input', 'a', 'logmix', 'mean', 'logstd',
                                'var_list'])
RNN_COMP_WITH_VAE = namedtuple("RNN_COMP_WITH_VAE",
                               ['logstd', 'mean', 'loss', 'pz'])


def build_vae(name, vae, na, z_size, seq_len, vae_lr, kl_tolerance):
  # used for later input tnesor
  a = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name + "_a")

  x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name=name + "_x")

  mean, logstd, z = vae.build_encoder(x, reuse=False)
  y = vae.build_decoder(z, reuse=False)
  tf_r_loss = -tf.reduce_sum(x * tf.log(y + 1e-8) +
                             (1. - x) * (tf.log(1. - y + 1e-8)), [1, 2, 3])
  tf_r_loss = tf.reduce_mean(tf_r_loss)

  tf_kl_loss = - 0.5 * tf.reduce_sum((1 + logstd - tf.square(mean)
                                      - tf.exp(logstd)), axis=1)
  tf_kl_loss = tf.reduce_mean(tf.maximum(tf_kl_loss, kl_tolerance * z_size))

  tf_vae_loss = tf_kl_loss + tf_r_loss
  vae_var_list = vae.get_variables()
  vae_fc_var_list = vae.get_fc_variables()

  vae_opt = tf.train.AdamOptimizer(vae_lr)
  vcomp = VAE_COMP(a=a, x=x, z=z, y=y, mean=mean, logstd=logstd,
                   r_loss=tf_r_loss, kl_loss=tf_kl_loss, loss=tf_vae_loss,
                   var_list=vae_var_list, fc_var_list=vae_fc_var_list,
                   train_opt=vae_opt)
  return vcomp


def build_mlp_vae(name, vae, na, z_size, seq_len, vae_lr, kl_tolerance):
  # used for later input tnesor
  a = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name + "_a")

  x = tf.placeholder(tf.float32, shape=[None, 4], name=name + "_x")

  mean, logstd, z = vae.build_encoder(x, reuse=False)
  y = vae.build_decoder(z, reuse=False)

  # (Lisheng) Use L2 loss temporally.
  tf_r_loss = 0.5 * tf.reduce_sum(tf.square(x - y), axis=1)
  tf_r_loss = tf.reduce_mean(tf_r_loss)

  tf_kl_loss = - 0.5 * tf.reduce_sum((1 + logstd - tf.square(mean)
                                      - tf.exp(logstd)), axis=1)
  tf_kl_loss = tf.reduce_mean(tf.maximum(tf_kl_loss, kl_tolerance * z_size))

  tf_vae_loss = tf_kl_loss + tf_r_loss
  vae_var_list = vae.get_variables()
  vae_fc_var_list = vae.get_fc_variables()

  vae_opt = tf.train.AdamOptimizer(vae_lr)
  vcomp = VAE_COMP(a=a, x=x, z=z, y=y, mean=mean, logstd=logstd,
                   r_loss=tf_r_loss, kl_loss=tf_kl_loss, loss=tf_vae_loss,
                   var_list=vae_var_list, fc_var_list=vae_fc_var_list,
                   train_opt=vae_opt)
  return vcomp


# Just build the structure.
def build_rnn(name, rnn, na, z_size, batch_size, seq_len):
  a = tf.placeholder(tf.float32, shape=[None, seq_len, na], name=name + "_a")
  rnn_z = tf.placeholder(dtype=tf.float32,
                         shape=[batch_size, seq_len, z_size],
                         name=name + "_z")

  input_x = tf.concat([rnn_z, a], axis=2)
  out_logmix, out_mean, out_logstd = rnn.build_model(input_x)

  rnn_var_list = rnn.get_variables()
  rcomp = RNN_COMP_WITH_OPT(a=a, z_input=rnn_z, logmix=out_logmix,
                            mean=out_mean, logstd=out_logstd,
                            var_list=rnn_var_list)

  return rcomp


# (Lisheng) Modified to be compatible with CartPole.
def process_z_with_vae(x, z, a, batch_size, seq_len, z_size, vae_type="conv"):
  # reshape and cut
  if vae_type == "conv":
    target_y = tf.reshape(x, (batch_size, seq_len + 1, 64, 64, 1))[:, 1:,
               ...]
    target_y = tf.reshape(target_y, (-1, 64, 64, 1))
  elif vae_type == "mlp":
    # Use cartpole's input size by default.
    target_y = tf.reshape(x, (batch_size, seq_len + 1, 4))[:, 1:, ...]
    target_y = tf.reshape(target_y, (-1, 4))
  else:
    raise Exception("The Vae type" + vae_type + "is not supported")

  input_z = tf.reshape(z, (batch_size, seq_len + 1, z_size))[:, :-1, :]
  input_z = tf.concat([input_z, a], axis=2)
  return input_z, target_y


# (Lisheng) Modified to be compatible with Cartpole.
def rnn_with_vae(vae, rnn, x, z, a, z_size, batch_size, seq_len, kl_tolerance,
                 vae_type):
  input_z, target_y = process_z_with_vae(x, z, a, batch_size, seq_len, z_size,
                                         vae_type)

  pz, mean, logstd = rnn.build_model(input_z, reuse=True)
  mean = tf.reshape(mean, [-1, z_size])
  logstd = tf.reshape(logstd, [-1, z_size])
  pz = tf.reshape(pz, [-1, z_size])
  py = vae.build_decoder(pz, reuse=True)  # -1, 64, 64, 1

  if vae_type == "conv":
    rnn_loss = tf.reduce_mean(get_lr_lossfunc(target_y, py))
  elif vae_type == "mlp":
    rnn_loss = 0.5 * tf.reduce_sum(tf.square(target_y - py), axis=1)
    rnn_loss = tf.reduce_mean(rnn_loss)
  else:
    raise Exception("The Vae type" + vae_type + "is not supported")

  rnn_kl_loss = get_kl2normal_lossfunc(mean, logstd)

  rnn_loss += tf.reduce_mean(tf.maximum(rnn_kl_loss, kl_tolerance * z_size))
  return rnn_loss, mean, logstd, pz


# Meta part.
# (Lisheng) Add a new argument to support MlpVAE
def build_rnn_with_vae(vae, rnn, vcomp, z_size, seq_len, batch_size,
                       kl_tolerance=0.5, vae_type="conv"):
  rnn_loss, mean, logstd, pz = rnn_with_vae(vae, rnn, vcomp.x, vcomp.z,
                                            vcomp.a,
                                            z_size, batch_size, seq_len,
                                            kl_tolerance,
                                            vae_type)
  rcomp = RNN_COMP_WITH_VAE(mean=mean, logstd=logstd, loss=rnn_loss, pz=pz)
  return rcomp


def get_transform_loss_with_y(vcomp, decoder, wrapper):
  y = decoder.build_decoder(vcomp.z, reuse=True)
  ty = wrapper.transform(y)
  transform_loss = -tf.reduce_sum(vcomp.x * tf.log(ty + 1e-8) +
                                  (1. - vcomp.x) * (tf.log(1. - ty + 1e-8)),
                                  [1, 2, 3])
  # TODO add one in the RNN's prediction error.
  transform_loss = tf.reduce_mean(transform_loss)
  return transform_loss, y


def get_transform_loss(vcomp, decoder, wrapper):
  loss, _ = get_transform_loss_with_y(vcomp, decoder, wrapper)
  return loss


def get_transform_loss_with_target(vcomp, decoder, target):
  y = decoder.build_decoder(vcomp.z, reuse=True)
  transform_loss = tf.reduce_mean(get_lr_lossfunc(target, y))
  return transform_loss


def get_predicted_transform_loss(vcomp, rcomp, decoder, wrapper, batch_size,
                                 seq_len):
  py = decoder.build_decoder(rcomp.pz, reuse=True)  # pz shape [None, 32]
  tpy = wrapper.transform(py)

  # target y
  y = tf.reshape(vcomp.x, (batch_size, seq_len + 1, 64, 64, 1))[:, 1:, ...]
  y = tf.reshape(y, (-1, 64, 64, 1))

  ptransform_loss = -tf.reduce_sum(y * tf.log(tpy + 1e-8) +
                                   (1. - y) * (tf.log(1. - tpy + 1e-8)),
                                   [1, 2, 3])
  ptransform_loss = tf.reduce_mean(ptransform_loss)
  return ptransform_loss


# TODO(lisheng) Use logistic regression loss
def get_predicted_transform_loss_with_target(rcomp, decoder, target):
  py = decoder.build_decoder(rcomp.pz, reuse=True)  # pz shape [None, 32]
  ptransform_loss = tf.reduce_mean(get_lr_lossfunc(target, py))
  return ptransform_loss


def build_vaes(n_tasks, na, z_size, seq_len, vrec_lr,
               kl_tolerance):
  vaes = []
  vcomps = []
  for i in range(n_tasks):
    vae = ConvVAE(name="vae%i" % i, z_size=z_size)
    vcomp = build_vae("vae%i" % i, vae, na, z_size, seq_len, vrec_lr,
                      kl_tolerance)
    vaes.append(vae)
    vcomps.append(vcomp)
  return vaes, vcomps


def build_rnns(n_tasks, rnn, vaes, vcomps, kl_tolerance):
  rcomps = []
  for i in range(n_tasks):
    vcomp = vcomps[i]
    vae = vaes[i]
    rcomp = build_rnn_with_vae(vae, rnn, vcomp, vae.z_size,
                               rnn.max_seq_len, rnn.batch_size,
                               kl_tolerance)
    rcomps.append(rcomp)
  return rcomps


def get_vmmd_losses(n_tasks, tcomp, vcomps, alpha, beta):
  target_mean = tf.stop_gradient(tf.reduce_mean(tcomp.mean, axis=0))
  target_logstd = tf.stop_gradient(tf.reduce_mean(tcomp.logstd, axis=0))
  mmd_losses = []
  for i in range(n_tasks):
    vcomp = vcomps[i]
    mean = tf.reduce_mean(vcomp.mean, axis=0)
    logstd = tf.reduce_mean(vcomp.logstd, axis=0)
    mmd_loss = tf.reduce_sum(alpha * tf.square(mean - target_mean) +
                             beta * tf.square(logstd - target_logstd))
    mmd_losses.append(mmd_loss)
  return mmd_losses


def get_rmmd_losses(n_tasks, tcomp, rcomps, alpha, beta):
  target_mean = tf.stop_gradient(tf.reduce_mean(tcomp.mean, axis=0))
  target_logstd = tf.stop_gradient(tf.reduce_mean(tcomp.logstd, axis=0))
  mmd_losses = []
  for i in range(n_tasks):
    rcomp = rcomps[i]
    mean = tf.reduce_mean(rcomp.mean, axis=0)
    logstd = tf.reduce_mean(rcomp.logstd, axis=0)
    mmd_loss = tf.reduce_sum(alpha * tf.square(mean - target_mean) +
                             beta * tf.square(logstd - target_logstd))
    mmd_losses.append(mmd_loss)
  return mmd_losses


def get_vae_rec_ops(n_tasks, vcomps, mmd_losses, w_mmd):
  vrec_ops = []
  for i in range(n_tasks):
    vcomp = vcomps[i]
    loss = vcomp.loss + mmd_losses[i] * w_mmd
    train_opt = vcomp.train_opt
    grads = train_opt.compute_gradients(loss, vcomp.var_list)
    rec_op = train_opt.apply_gradients(grads, name="vae_train_op_%i" % i)
    vrec_ops.append(rec_op)
  return vrec_ops


def get_vae_pred_ops(n_tasks, vcomps, rnn_losses):
  vpred_ops = []
  tf_vpred_lrs = []
  for i in range(n_tasks):
    vcomp = vcomps[i]
    tf_vpred_lr = tf.placeholder(tf.float32, shape=[])  # learn from vr
    # TODO(lisheng) Consider RMSPropGrad.
    vpred_opt = tf.train.AdamOptimizer(tf_vpred_lr, name="vpred_opt%i"
                                                         % i)
    gvs = vpred_opt.compute_gradients(rnn_losses[i], vcomp.var_list)
    vpred_op = vpred_opt.apply_gradients(gvs, name='vpred_op%i' % i)
    vpred_ops.append(vpred_op)
    tf_vpred_lrs.append(tf_vpred_lr)
  return vpred_ops, tf_vpred_lrs
