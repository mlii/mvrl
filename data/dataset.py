import numpy as np

from utils import onehot_actions


class DataSet(object):
  def __init__(self, seq_len, na, data_dir, fns, data_type="img"):
    self.data_dir = data_dir
    self.fns = np.array(fns)
    self.seq_len = seq_len
    self.na = na
    self.n = len(self.fns)
    self.ids = np.arange(self.n)
    self.i = 0
    np.random.shuffle(self.ids)
    self.data_type = data_type

  def random_batch(self, batch_size, return_ids=False):
    sample_fns = self.get_sample_fns(batch_size)
    return self.get_batch(sample_fns, return_ids)

  def get_sample_fns(self, batch_size):
    indices = self.ids[self.i:self.i + batch_size]
    sample_fns = self.fns[indices]

    self.i += batch_size
    if self.i >= self.n:
      np.random.shuffle(self.ids)
      self.i = 0

    nb = len(sample_fns)
    if nb < batch_size:
      sample_fns = np.concatenate(
        [sample_fns, self.get_sample_fns(batch_size - nb)])
    return sample_fns

  def get_batch(self, fns, return_ids=False):
    obs = []
    a = []
    ids = []
    for fn in fns:
      tobs, ta, idx = self.load_sample_data(fn, self.seq_len, self.na)
      obs.append(tobs)
      a.append(ta)
      ids.append(idx)

    obs = np.array(obs)
    if self.data_type == "img":
      obs = np.expand_dims(obs, axis=-1) / 255.

    a = np.array(a)
    if not return_ids:
      return obs, a
    else:
      return obs, a, fns, np.array(ids)

  def get_batch_with_ids(self, fns, ids):
    obs = []
    a = []

    for i, fn in enumerate(fns):
      tobs, ta, _ = self.load_sample_data(fn, self.seq_len,
                                          self.na, ids[i])
      obs.append(tobs)
      a.append(ta)

    obs = np.array(obs)
    if self.data_type == "img":
      obs = np.expand_dims(obs, axis=-1) / 255.
    a = np.array(a)
    return obs, a

  def load_sample_data(self, fn, seq_len, na, idx=None):
    raw_data = np.load(self.data_dir + '/' + fn)
    n = len(raw_data["obs"])
    # n = len(raw_data["action"])
    # otherwise
    if idx is None:
      idx = np.random.randint(0,
                              n - seq_len)  # the final one won't be taken
    a = raw_data["action"][idx:idx + seq_len + 1]  # sample one more.
    obs = raw_data["obs"][idx:idx + seq_len + 1]  # sample one more
    oh_a = onehot_actions(a, na)
    return obs, oh_a, idx


class DatasetManager(object):
  def __init__(self, datasets):
    self.datasets = datasets

  def random_batch(self, batch_size_per_task):
    obs_list, a_list = [], []
    for d in self.datasets:
      obs, a = d.random_batch(batch_size_per_task)
      obs = obs.reshape((-1,) + obs.shape[2:])
      obs_list.append(obs)
      a_list.append(a)
    return obs_list, a_list


class DatasetManagerCor(object):
  def __init__(self, dataset, wrapper):
    self.dataset = dataset
    self.wrapper = wrapper

  def random_batch(self, batch_size_per_task):
    obs, a = self.dataset.random_batch(batch_size_per_task)
    obs = obs.reshape((-1,) + obs.shape[2:])
    obs_list = [obs, self.wrapper.data_transform(obs)]
    a_list = [a, a]
    return obs_list, a_list


# Manage multiple datasets.
class DatasetMixManager(object):
  def __init__(self, datasets):
    self.datasets = datasets

  def random_batch(self, batch_size_per_task):
    obs_list, a_list = [], []
    for d in self.datasets:
      obs, a = d.random_batch(batch_size_per_task)
      obs = obs.reshape((-1,) + obs.shape[2:])
      obs_list.append(obs)
      a_list.append(a)
    return obs_list, a_list

  def random_batch_with_targets(self, batch_size_per_task):
    obs_list, a_list = [], []
    target_list = []

    obs, a, sample_fns, sample_ids = self.datasets[0].random_batch(
      batch_size_per_task,
      return_ids=True)
    obs = obs.reshape((-1,) + obs.shape[2:])
    obs_list.append(obs)
    a_list.append(a)

    for d in self.datasets[1:]:
      obs, a = d.random_batch(batch_size_per_task)
      obs = obs.reshape((-1,) + obs.shape[2:])
      obs_list.append(obs)
      a_list.append(a)

      # Get the corresponding data to the first one.
      obs, _ = d.get_batch_with_ids(sample_fns, sample_ids)
      obs = obs.reshape((-1,) + obs.shape[2:])
      target_list.append(obs)
    return obs_list, a_list, target_list

  def random_cor_batch(self, batch_size_per_task):
    obs_list, a_list = [], []

    obs, a, sample_fns, sample_ids = self.datasets[0].random_batch(
      batch_size_per_task,
      return_ids=True)

    obs = obs.reshape((-1,) + obs.shape[2:])
    obs_list.append(obs)
    a_list.append(a)

    for d in self.datasets[1:]:
      obs, a = d.get_batch_with_ids(sample_fns, sample_ids)
      obs = obs.reshape((-1,) + obs.shape[2:])
      obs_list.append(obs)
      a_list.append(a)
    return obs_list, a_list, obs_list[1:]


def get_dm(wrapper, seq_len, na, data_dir, fns, cor=False):
  if not cor:
    dataset1 = DataSet(seq_len, na, data_dir, fns)
    dataset2 = DataSet(seq_len, na, data_dir, fns)
    datasets = [dataset1, wrapper(dataset2)]
    dm = DatasetManager(datasets)  # sample from this one.
  else:
    dataset = DataSet(seq_len, na, data_dir, fns)
    dm = DatasetManagerCor(dataset, wrapper)
  return dm


def get_mixed_dm(seq_len, na, data_dirs, fns_list, data_types):
  if set(fns_list[0]) != set(fns_list[1]):
    print("Warning! You can't get targets using DatasetMixManager.")
  dataset1 = DataSet(seq_len, na, data_dirs[0], fns_list[0], data_types[0])
  dataset2 = DataSet(seq_len, na, data_dirs[1], fns_list[1], data_types[1])
  dm = DatasetMixManager([dataset1, dataset2])
  return dm
