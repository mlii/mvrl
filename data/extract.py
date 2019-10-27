'''
saves ~ 200 episodes generated from a random policy
'''

import os
import random

import gym
import numpy as np
from data.model import make_model

MAX_FRAMES = 1000  # max length of carracing
MAX_TRIALS = 200  # just use this to extract one trial.
ENV_NAME = "Pong-v0"

render_mode = False  # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
  os.makedirs(DIR_NAME)

print("Directory built")
model = make_model(load_model=False, env_name=ENV_NAME)

print("Make Model")
total_frames = 0
model.make_env(render_mode=render_mode)

obs = model.env.reset()
action = model.env.action_space.sample()
recording_obs = np.array([obs for i in range(MAX_FRAMES)], dtype="uint8")
recording_action = np.array([action for i in range(MAX_FRAMES)])

for trial in range(MAX_TRIALS):  # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2 ** 31 - 1)
    filename = DIR_NAME + "/" + str(random_generated_int) + ".npz"

    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)

    # random policy
    model.init_random_model_params(stdev=np.random.rand() * 0.01)

    model.reset()
    obs = model.env.reset()  # pixels

    for frame in range(MAX_FRAMES):
      if render_mode:
        model.env.render("human")
      else:
        model.env.render("rgb_array")

      recording_obs[frame] = obs

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z, epsilon=0.5)

      recording_action[frame] = action
      obs, reward, done, info = model.env.step(action)

      if done:
        break

    total_frames += (frame + 1)
    print("dead at", frame + 1, "total recorded frames for this worker", total_frames)
    np.savez_compressed(filename, obs=recording_obs[:frame + 1], action=recording_action[:frame + 1])
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
