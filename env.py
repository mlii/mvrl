import cv2
import gym
import numpy as np
from gym import spaces

SCREEN_X = 64
SCREEN_Y = 64


class PongBinary(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(64, 64, 1), dtype=np.uint8)

  def observation(self, frame):
    frame = frame[35:195, :, 0]
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 255
    frame = frame[::2, ::2]
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    return frame


# Borrowed from the universe-starter-agent, openai baselines

class AtariRescale64x64(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(64, 64, 1), dtype=np.uint8)

  def observation(self, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    return frame


class AtariRescaleClip64x64(gym.ObservationWrapper):
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(64, 64, 1), dtype=np.uint8)

  def observation(self, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (64, 84), interpolation=cv2.INTER_AREA)
    frame = frame[14:78, ...]
    return frame


class NoopResetEnv(gym.Wrapper):
  def __init__(self, env, noop_max=30):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

  def reset(self, **kwargs):
    """ Do no-op action for a number of steps in [1, noop_max]."""
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

  def step(self, ac):
    return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    self._skip = skip

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2: self._obs_buffer[0] = obs
      if i == self._skip - 1: self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class SkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    self._skip = skip

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      if done:
        break

    return obs, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    """Bin reward to {+1, 0, -1} by its sign."""
    return np.sign(reward)


def make_atari(env_id, noop_max=3, clip_frame=True):
  env = gym.make(env_id)
  env = NoopResetEnv(env, noop_max=noop_max)
  env = SkipEnv(env, skip=4)
  # env = MaxAndSkipEnv(env, skip=4)
  if env_id == "PongNoFrameskip-v4":
    env = PongBinary(env)
  else:
    if clip_frame:
      env = AtariRescaleClip64x64(env)
    else:
      env = AtariRescale64x64(env)
  env = ClipRewardEnv(env)
  return env


class CartPoleWrapper(object):
  def __init__(self):
    self.env = gym.make("CartPole-v0")

  def render(self, rate=10):
    if self.env.env.state is None: return None
    import cv2

    screen_width = 600
    screen_height = 400

    world_width = self.env.env.x_threshold * 2
    scale = screen_width / world_width
    carty = 100  # TOP OF CART
    polewidth = 10.0
    polelen = scale * (2 * self.env.env.length)
    cartwidth = 50.0
    cartheight = 30.0
    axleoffset = cartheight / 4.0

    img = np.empty((screen_height, screen_width, 3), dtype=np.uint8)
    img[...] = 255

    x = self.env.env.state
    cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    # self.carttrans.set_translation(cartx, carty)
    # self.poletrans.set_rotation(-x[2])

    color_cart = (0, 0, 0)
    cv2.rectangle(img,
                  (int(cartx - cartwidth / 2), screen_height - int(carty - cartheight / 2)),
                  (int(cartx + cartwidth / 2), screen_height - int(carty + cartheight / 2)),
                  color_cart, -1, lineType=cv2.LINE_AA)

    # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    # pole.set_color(.8, .6, .4)
    # self.poletrans = rendering.Transform(translation=(0, axleoffset))
    # pole.add_attr(self.poletrans)
    # pole.add_attr(self.carttrans)
    # self.viewer.add_geom(pole)
    color_pole = (204, 153, 102)
    pole = np.array([(cartx - polewidth / 2, screen_height - carty),
                     (cartx - polewidth / 2, screen_height - (carty + polelen)),
                     (cartx + polewidth / 2, screen_height - (carty + polelen)),
                     (cartx + polewidth / 2, screen_height - carty)], np.int32)
    # cv2.fillConvexPoly(img, pole, color_pole)
    apex = cartx, screen_height - (carty + axleoffset)
    # pole += np.int32(apex)
    pole = np.array([pole])  # use an array of array of points
    rotate_angle = -x[2] / 3.141592 * 180
    pole_m = cv2.getRotationMatrix2D(center=apex, angle=rotate_angle, scale=1)
    rotated_pole = cv2.transform(pole, pole_m)
    cv2.fillConvexPoly(img, rotated_pole, color_pole)

    # self.axle = rendering.make_circle(polewidth/2)
    # self.axle.add_attr(self.poletrans)
    # self.axle.add_attr(self.carttrans)
    # self.axle.set_color(.5,.5,.8)
    # self.viewer.add_geom(self.axle)
    color_axle = (127.5, 127.5, 204.0)
    cv2.circle(img, (int(cartx), screen_height - int(carty + polewidth / 2)), int(polewidth / 2), color_axle, -1,
               lineType=cv2.LINE_AA)

    # self.track = rendering.Line((0, carty), (screen_width, carty))
    # self.track.set_color(0, 0, 0)
    # self.viewer.add_geom(self.track)
    # self._pole_geom = pole

    # cv2.imshow('cartpole', img)
    # cv2.waitKey(rate)

    return img

  def reset(self):
    obs = self.env.reset()
    # img_obs = self.env.render(mode="rgb_array")
    img_obs = self.render()
    obs = [obs, img_obs]
    return obs

  # Every step. Get resized image observation
  # and numerical data together.
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    # img_obs = self.env.render(mode="rgb_array")
    img_obs = self.render()
    obs = [obs, img_obs]
    return obs, reward, done, info

  def seed(self, s):
    self.env.seed(s)

  @property
  def action_space(self):
    return self.env.action_space


# def make_cartpole(seed=-1):
#     env = CartPoleWrapper()
#     if seed >= 0:
#         env.seed(seed)
#     return env

# useless render mode
def make_env(env_name, seed=-1):
  if env_name == "CartPole-v0":
    env = CartPoleWrapper()
  else:
    env = make_atari(env_name)

  if (seed >= 0):
    env.seed(seed)
  return env
