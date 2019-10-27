from abc import ABC, abstractmethod, abstractclassmethod

import numpy as np
import tensorflow as tf


class WrapperBase(ABC):
  def __init__(self, dataset):
    self.dataset = dataset

  @abstractmethod
  def random_batch(self, batch_size):
    pass

  @abstractclassmethod
  def transform(cls, x):
    pass

  @abstractclassmethod
  def data_transform(cls, x):
    pass


class DatasetTransposeWrapper(WrapperBase):
  def random_batch(self, batch_size):
    obs, a = self.dataset.random_batch(batch_size)
    obs = np.transpose(obs, [0, 1, 3, 2, 4])
    return obs, a

  @classmethod
  def transform(cls, x):
    # x [None, 64, 64, 1]
    x = tf.transpose(x, [0, 2, 1, 3])
    return x

  @classmethod
  def data_transform(cls, x):
    # x [None, 64, 64, 1]
    x = np.transpose(x, [0, 2, 1, 3])
    return x


class DatasetSwapWrapper(WrapperBase):
  def random_batch(self, batch_size):
    obs, a = self.dataset.random_batch(batch_size)
    obs = obs[:, :, :, ::-1, :]  # left <-> right
    return obs, a

  @classmethod
  def transform(cls, x):
    x = x[:, :, ::-1, :]
    return x

  @classmethod
  def data_transform(cls, x):
    x = x[:, :, ::-1, :]
    return x


class DatasetHorizontalConcatWrapper(WrapperBase):
  def random_batch(self, batch_size):
    obs, a = self.dataset.random_batch(batch_size)
    sub_obs = np.split(obs, 2, axis=3)
    obs = np.concatenate(sub_obs[::-1], axis=3)
    return obs, a

  @classmethod
  def transform(cls, x):
    xs = tf.split(x, 2, axis=2)
    x = tf.concat(xs[::-1], axis=2)
    return x

  @classmethod
  def data_transform(cls, x):
    xs = np.split(x, 2, axis=2)
    x = np.concatenate(xs[::-1], axis=2)
    return x


class DatasetColorWrapper(WrapperBase):
  def random_batch(self, batch_size):
    obs, a = self.dataset.random_batch(batch_size)
    return 1.0 - obs, a

  @classmethod
  def transform(cls, x):
    return 1.0 - x

  @classmethod
  def data_transform(cls, x):
    return 1.0 - x


class WrapperFactory(object):
  Wrapper_dict = {
    "transposed": DatasetTransposeWrapper,
    "mirror": DatasetSwapWrapper,
    "h-swapped": DatasetHorizontalConcatWrapper,
    "inverse": DatasetColorWrapper
  }

  @classmethod
  def get_wrapper(cls, item):
    return cls.Wrapper_dict.get(item, None)
