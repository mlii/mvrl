import os
import pickle

import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp

logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))


def wrap(wrapper, s):
  if wrapper is None:
    return s
  else:
    return wrapper.data_transform(s)


def saveToFlat(var_list, param_pkl_path):
  # get all the values
  var_values = np.concatenate(
    [v.flatten() for v in tf.get_default_session().run(var_list)])
  pickle.dump(var_values, open(param_pkl_path, "wb"))


def load_from_file(param_pkl_path):
  with open(param_pkl_path, 'rb') as f:
    params = pickle.load(f)
  return params.astype(np.float32)


def loadFromFlat(var_list, param_pkl_path):
  flat_params = load_from_file(param_pkl_path)
  print("the type of the parameters stored is ", flat_params.dtype)
  shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
  total_size = np.sum([int(np.prod(shape)) for shape in shapes])
  theta = tf.placeholder(tf.float32, [total_size])
  start = 0
  assigns = []
  for (shape, v) in zip(shapes, var_list):
    size = int(np.prod(shape))
    # print(v.name)
    assigns.append(
      tf.assign(v, tf.reshape(theta[start:start + size], shape)))
    start += size
  op = tf.group(*assigns)
  tf.get_default_session().run(op, {theta: flat_params})


def check_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)


def reset_graph():
  # global variables.
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()


def create_vae_dataset(filelist, N=10000,
                       M=1000):  # N is 10000 episodes, M is number of timesteps
  data = np.zeros((M * N, 64, 64, 1), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    raw_data = np.expand_dims(raw_data, axis=-1)
    l = len(raw_data)
    if (idx + l) > (M * N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx + l] = raw_data
    idx += l
    if ((i + 1) % 100 == 0):
      print("loading file", i + 1)

  if len(data) == M * N and idx < M * N:
    data = data[:idx]
  return data


def lognormal(y, mean, logstd):
  return -0.5 * ((y - mean) / np.exp(logstd)) ** 2 - logstd - logSqrtTwoPI


def tf_lognormal(y, mean, logstd):
  return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI


def get_lossfunc(logmix, mean, logstd, y):
  v = logmix + tf_lognormal(y, mean, logstd)
  v = tf.reduce_logsumexp(v, 1, keepdims=True)
  return -tf.reduce_mean(v)


def get_l2_lossfunc(mean, logstd, target_mean, target_logstd):
  return (tf.square(mean - target_mean) + tf.square(logstd - target_logstd))


def get_kl_lossfunc(mean, logstd, target_mean, target_logstd):
  return tf.reduce_sum(logstd - target_logstd + (
      tf.exp(2 * target_logstd) + tf.square(target_mean - mean)) / 2 / tf.exp(
    2 * logstd) - 0.5, axis=1)


def get_kl2normal_lossfunc(mean, logstd):
  tf_kl_loss = - 0.5 * tf.reduce_sum(
    (1 + logstd - tf.square(mean) - tf.exp(logstd)), axis=1)
  return tf_kl_loss


def get_lr_lossfunc(y, py):
  return -tf.reduce_sum(
    y * tf.log(py + 1e-8) + (1. - y) * (tf.log(1. - py + 1e-8)), [1, 2, 3])


def neg_likelihood(logmix, mean, logstd, y):
  v = logmix + lognormal(y, mean, logstd)
  v = logsumexp(v, 1, keepdims=True)
  return -np.mean(v)


def onehot_actions(actions, na):
  actions = actions.astype(np.uint8)
  l = len(actions)
  oh_actions = np.zeros((l, na))
  oh_actions[np.arange(l), actions] = 1
  return oh_actions


def onehot_action(action, na):
  oh_action = np.zeros((na,))
  oh_action[action] = 1
  return oh_action


def pad_num(n):
  s = str(n)
  return '0' * (4 - len(s)) + s


def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1


def sample_z(logmix, mean, logstd, l, T=1):
  if T == 1:
    logmix2 = np.copy(logmix) / T
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(l, 1)
  else:
    logmix2 = np.copy(logmix)

  mixture_idx = np.zeros(l)
  chosen_mean = np.zeros(l)
  chosen_logstd = np.zeros(l)

  for j in range(l):
    idx = get_pi_idx(np.random.rand(), logmix2[j])
    mixture_idx[j] = idx
    chosen_mean[j] = mean[j][idx]
    chosen_logstd[j] = logstd[j][idx]

  rand_gaussian = np.random.randn(l) * np.sqrt(T)
  next_z = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
  return next_z


def print_var_name(vars):
  print([v.name for v in vars])


def iter_cost(prefix, cost_array):
  s = "".join(["%s%i: %.2f, " % (prefix, i, cost) for i, cost in
               enumerate(cost_array)])
  return s


def get_output_log(step, rpred_lr, vcost_array, rcost_array, tcost_array,
                   ptcost_array):
  output_log = "step: %d, lr: %.6f, " % (step, rpred_lr)
  # r cost: p, v cost: r, t cost: t, pt cost: pt
  output_log += iter_cost("r", vcost_array)
  output_log += iter_cost("p", rcost_array)
  output_log += iter_cost("t", tcost_array)
  output_log += iter_cost("pt", ptcost_array)
  output_log += "\n"
  return output_log
