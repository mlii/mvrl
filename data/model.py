import json

import numpy as np

from env import make_env

MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3  # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH


# controls whether we concatenate (z, c, h), etc for features used for car.

def make_model(load_model=True, env_name="Pong-v0"):
  # can be extended in the future.
  model = Model(load_model=load_model, env_name=env_name)
  return model


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def relu(x):
  return np.maximum(x, 0)


def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)


def passthru(x):
  return x


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


def sample(p):
  return np.argmax(np.random.multinomial(1, p))


class Model:
  ''' simple one layer model for car racing '''

  def __init__(self, load_model=True, env_name="Pong-v0", render_mode=False):
    self.env_name = env_name

    self.rnn_mode = True

    self.na = 2
    self.input_size = 4
    self.init_controller()

    self.render_mode = False

  # INIT The Controller After the enviroment Creation.
  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)
    self.na = self.env.action_space.n  # discrete by default.

  def init_controller(self):
    if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, self.na)  # pong. Modify later.
      self.bias_output = np.random.randn(self.na)
      self.param_count = (self.input_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.na
    else:
      self.weight = np.random.randn(self.input_size, self.na)
      self.bias = np.random.randn(self.na)
      self.param_count = (self.input_size + 1) * self.na

  def get_action(self, z, epsilon=0.0):
    h = z
    if np.random.rand() < epsilon:
      action = np.random.randint(0, self.na)
    else:
      if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
        h = np.maximum(np.dot(h, self.weight_hidden) + self.bias_hidden, 0)
        action = np.argmax(np.dot(h, self.weight_output) + self.bias_output)
      else:
        action = np.argmax(np.dot(h, self.weight) + self.bias)

    action = np.random.randint(0, self.na)

    oh_action = np.zeros(self.na)
    oh_action[action] = 1

    return action

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size + 1) * self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:self.na]
      self.weight_output = params_2[self.na:].reshape(self.hidden_size, self.na)
    else:
      self.bias = np.array(model_params[:self.na])
      self.weight = np.array(model_params[self.na:]).reshape(self.input_size, self.na)

  def load_model(self, filename):
    with open(filename) as f:
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0])  # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    # return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count) * stdev  # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
