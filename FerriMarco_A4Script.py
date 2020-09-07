import numpy as np

# %tensorflow_version 1.x
import tensorflow as tf

import os
os.environ.setdefault('PATH', '')

from collections import deque
import gym
from gym import spaces

import cv2
cv2.ocl.setUseOpenCL(False)

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

# ----------------------------------------------------

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
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

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

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env



################################################################
####################     DEFINED BY ME     #####################

def wrap_atari_deepmind(environment_name, clipped_reward):
    atari = make_atari(environment_name)
    wrapper = wrap_deepmind(atari, 
                            episode_life=True, 
                            clip_rewards=clipped_reward, 
                            frame_stack=True, 
                            scale=True)
    return wrapper

# ----------------------------------------------------

def generate_network(scope, learning_rate, decay, gamma, actions_number):

    with tf.variable_scope(scope):

        X = tf.placeholder(tf.float32, shape=(None, 84, 84, 4), name = "X")
        
        W_init = tf.variance_scaling_initializer()
        b_init = tf.zeros_initializer()

        # conv layer 1
        W_c1 = tf.get_variable(name = "W_c1", shape=[8, 8, 4, 32], initializer = W_init) # 32 filters 8x8 on 3 channels
        b_c1 = tf.get_variable(name = "b_c1", shape=[32], initializer = b_init)
        A_c1 = tf.nn.relu(tf.nn.conv2d(X, W_c1, strides=[1, 4, 4, 1], padding= 'SAME') + b_c1)

        # conv layer 2
        W_c2 = tf.get_variable(name = "W_c2", shape=[4, 4, 32, 64], initializer = W_init) # 64 filters 4x4 on 32 channels
        b_c2 = tf.get_variable(name = "b_c2", shape=[64], initializer = b_init)
        A_c2 = tf.nn.relu(tf.nn.conv2d(A_c1, W_c2, strides=[1, 2, 2, 1], padding= 'SAME') + b_c2)

        # conv layer 3
        W_c3 = tf.get_variable(name = "W_c3", shape=[3, 3, 64, 64], initializer = W_init) # 64 filters 3x3 on 64 channels
        b_c3 = tf.get_variable(name = "b_c3", shape=[64], initializer = b_init)
        A_c3 = tf.nn.relu(tf.nn.conv2d(A_c2, W_c3, strides=[1, 1, 1, 1], padding='SAME') + b_c3)
        
        # flatten
        A_c3_shapes = A_c3.shape.as_list()
        A_c3_flatten_shape = np.product(A_c3_shapes[1:])
        A_pool_flat = tf.reshape(A_c3, [-1, A_c3_flatten_shape]) # ? x A_c3_flatten_shape TODO shape

        # fc layer 1
        W_fc1 = tf.get_variable(name = "W_fc1", shape=[A_c3_flatten_shape, 512], initializer = W_init) # TODO shape
        b_fc1 = tf.get_variable(name = "b_fc1", shape=[512], initializer = b_init)
        A_fc1 = tf.nn.relu(tf.matmul(A_pool_flat, W_fc1) + b_fc1) # ? x 512

        # fc layer 2
        W_fc2 = tf.get_variable(name = "W_fc2", shape=[512, actions_number], initializer = W_init)
        b_fc2 = tf.get_variable(name = "b_fc2", shape=[actions_number], initializer=b_init)

        # output
        Z = tf.matmul(A_fc1, W_fc2) + b_fc2 # ? x actions_number
        
        # --------------

        omega = tf.placeholder(tf.bool, [None, 1], name = "omega")
        next_q = tf.placeholder(tf.float32, [None, actions_number], name = "next_q")
        reward = tf.placeholder(tf.float32, [None, 1], name = "reward")
        action = tf.placeholder(tf.int32, [None, 1], name = "action")

        y = tf.where(omega, reward, reward + gamma * tf.reduce_max(next_q, axis=1, keep_dims=True))
        Z_gather = tf.gather(Z, action, axis=1, batch_dims=1)

        loss = tf.reduce_sum(tf.squared_difference(y, Z_gather))

        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = decay)
        train = optimizer.minimize(loss)

    return loss, train, Z

# ----------------------------------------------------

def copy_network_parameters(source, destination):
    vars_on = tf.trainable_variables(scope=source)
    vars_off = tf.trainable_variables(scope=destination)

    operations = [tf.assign(ref=dest, value=src) for src, dest in zip(vars_on, vars_off)]
    return tf.group(operations)

# ----------------------------------------------------

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = np.zeros((size, 5), dtype=np.object)
        self.size = size
        self.__count = 0

    def add(self, transition):
        self.buffer[self.__count % self.size] = np.array(transition)
        self.__count += 1
    
    def get(self, number_of_transition):
        if number_of_transition > self.__count:
            number_of_transition = self.__count
        source = self.buffer[:self.__count] if self.__count < self.size else self.buffer
        indices = np.random.permutation(len(source))[:number_of_transition]
        return source[indices]
        
# ----------------------------------------------------

import random
import pandas as pd
from datetime import timedelta, datetime

def training(path_save, N, M, eps, gamma, B, lr, n, C, decay, eps_decay_min, eps_decay_step,
            moving_avg_window, eval_steps, eval_eps, eval_plays, eval_episodes):

    env_id = 'BreakoutNoFrameskip-v4'
    env = wrap_atari_deepmind(env_id, True)
    env_eval = wrap_atari_deepmind(env_id, False)
    # img = plt.imshow(env.render('rgb_array')) # for Colab

    # Useful Parameters
    actions_number = env.action_space.n
    eps_reduction = (eps - eps_decay_min) / eps_decay_step # how much eps should be decreased over time steps
    replay_buffer = ReplayBuffer(M)

    # Newtork Setup
    tf.reset_default_graph()
    cnn_online_loss, cnn_online_train, cnn_online_out = generate_network('online', lr, decay, gamma, actions_number)
    _, _, cnn_offline_out = generate_network('offline', lr, decay, gamma, actions_number)
    
    session = tf.Session()
    saver = tf.train.Saver()

    # TensoarBoard Config
    tb_path =  '/tmp/gradient_descent'
    os.makedirs(tb_path, exist_ok=True)
    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tb_path, session.graph)
    
    session.run(tf.global_variables_initializer())
    copy_operator = copy_network_parameters('online', 'offline')
    session.run(copy_operator)

    # Metrics
    info_basic = []
    info_loss = []
    info_score_eval = []
    info_score_train = []

    # Training
    step = 0
    episode = 0
    while step < N:
        observation = env.reset()
        done = False
        episode += 1
        
        if episode % 100 == 0:
          print(f'EPISODE {episode}')

        while(not done):
            # env.render()
            # img.set_data(env.render('rgb_array')) # for Colab
            # display.display(plt.gcf()) # for Colab
            # display.clear_output(wait=True) # for Colab
            
            if random.random() < (1 - eps):
                q = session.run(cnn_online_out, { "online/X:0": observation[np.newaxis,...] })
                action = np.argmax(q)
            else:
                action = env.action_space.sample()

            observation_next, reward, done, info = env.step(action)
            transition = [observation, action, reward, observation_next, done]
            replay_buffer.add(transition)

            step += 1
            if step % 10000 == 0:
                print(f'STEP {step}')
            
            if step >= M: # no training for the first M steps
                if step % n == 0:
                    batch = replay_buffer.get(B)
                    b_obs = np.array(list(map(lambda item: np.array(item), batch[:,0]))) # extract images from LazyFrames
                    b_action = np.array(batch[:,1][..., np.newaxis])
                    b_reward = np.array(batch[:,2][..., np.newaxis])
                    b_obs_next = np.array(list(map(lambda item: np.array(item), batch[:,3]))) # extract images from LazyFrames
                    b_done = np.array(batch[:,4][..., np.newaxis])

                    next_q = session.run(cnn_offline_out, { "offline/X:0": b_obs_next })
                    _, l = session.run([cnn_online_train, cnn_online_loss], 
                                        { "online/X:0": b_obs, "online/action:0": b_action, "online/reward:0": b_reward, 
                                        "online/omega:0": b_done, "online/next_q:0": next_q })
                    info_loss.append([episode, step, l])

                if step % C == 0:
                    session.run(copy_operator)
            
            if step < eps_decay_step:
                eps -= eps_reduction

            observation = observation_next
            # print(f'EPISODE {episode} \t STEP {step} \t\t Reward: {reward}')
            info_basic.append([episode, step, reward])

            # Return during Evaluation
            if(step % eval_steps == 0):
                print('Evalutation...')
                score = 0
                for eval_play in range(eval_plays):
                    for eval_episode in range(eval_episodes):
                        eval_observation = env_eval.reset()
                        eval_done = False
                        while not eval_done:
                            # env_eval.render()
                            if random.random() < (1 - eval_eps):
                                eval_q = session.run(cnn_online_out, { "online/X:0": eval_observation[np.newaxis,...] })
                                eval_action = np.argmax(eval_q)
                            else:
                                eval_action = env_eval.action_space.sample()
                            
                            eval_observation, eval_reward, eval_done, eval_info = env_eval.step(eval_action)
                            score += eval_reward

                score /= eval_plays
                print(f'STEP {step} \t SCORE: ', score)
                info_score_eval.append(score)
                
                # Checkpoint
                print(f"Saving checkpoint... (step {step})")
                saver.save(session, os.path.join(path_save, f'model{step}.ckpt'), step)

    # Return during Training
    info_basic = np.array(info_basic)
    info_basic_df = pd.DataFrame({'episode': info_basic[:, 0], 'step': info_basic[:, 1], 'reward': info_basic[:, 2]})
    return_per_episode = info_basic_df.groupby(['episode']).sum()['reward']
    info_score_train = return_per_episode.rolling(window=moving_avg_window, center=True).mean() # moving average
    
    # Saving video
    print("Saving video...")
    env_video = gym.wrappers.Monitor(env_eval, path_save, video_callable=lambda _: True, force=True)
    for _ in range(10):
        video_observation = env_video.reset()
        video_done = False

        while(not video_done):
            # env_video.render()

            if random.random() < (1 - eval_eps):
                video_q = session.run(cnn_online_out, { "online/X:0": video_observation[np.newaxis,...] })
                video_action = np.argmax(video_q)
            else:
                video_action = env_video.action_space.sample()
            
            video_observation, video_reward, video_done, video_info = env_video.step(video_action)

    # Close and return
    env_video.close()
    env_eval.close()
    env.close()
    session.close()
    print("Training finished")

    return np.array(info_basic), np.array(info_loss), np.array(info_score_train), np.array(info_score_eval)

# ----------------------------------------------------

# path_save = f'drive/My Drive/_ USI/Deep Learning Lab/Activity8/checkpoints/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
path = f'checkpoints/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
os.makedirs(path, exist_ok=True)

rewards, losses, scores_training, scores_eval = training(
        path_save=path,
        N=2000000, 
        M=10000, 
        eps=1, 
        gamma=0.99, 
        B=32, 
        lr=0.0001, 
        n=4, 
        C=10000, 
        decay=0.99, 
        eps_decay_min=0.1, 
        eps_decay_step=1000000, 
        moving_avg_window=30,
        eval_steps=100000,
        eval_eps=0.001,
        eval_plays=30,
        eval_episodes=5
    )

np.save(os.path.join(path, 'numpy_rewards'), rewards)
np.save(os.path.join(path, 'numpy_losses'), losses)
np.save(os.path.join(path, 'numpy_scores_eval'), scores_eval)
np.save(os.path.join(path, 'numpy_scores_training'), scores_training)

# ----------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt

colors = ['#616BB0', '#74C49D', '#FFFF00', '#B02956', '#B3BAFF']

# Total loss
ys = losses[:,2]
xs = np.arange(len(ys))
plt.plot(xs, ys, '-', c=colors[0], label='Temporal difference error')
plt.legend()
# plt.xticks(np.arange(0, len(ys)+batches_number, step=batches_number), ('0', '1', '2', '3', '4', '5'))
plt.show()

# Train score
ys = scores_training
xs = np.arange(len(ys))
plt.plot(xs, ys, '-', c=colors[1], label='Training scores')
plt.legend()
# plt.xticks(np.arange(0, len(ys)+batches_number, step=batches_number), ('0', '1', '2', '3', '4', '5'))
plt.show()

# Eval score
ys = scores_eval
xs = np.arange(len(ys))
plt.plot(xs, ys, '-', c=colors[3], label='Evaluation scores')
plt.legend()
# plt.xticks(np.arange(0, len(ys)+batches_number, step=batches_number), ('0', '1', '2', '3', '4', '5'))
plt.show()