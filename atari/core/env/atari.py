from collections import deque

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


class AtariEnvWrapper(object):
    """ Atari Environment Wrapper
    
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """

    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        # agent avaiable actions
        # such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(gym_env.action_space.n)
        # screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.state_buffer = deque()


    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self._get_preprocessing_frame(x_t)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)

        return s_t


    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer.

        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self._get_preprocessing_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.action_repeat, 84, 84))
        s_t1[:self.action_repeat-1,:] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        # pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info


    def _get_preprocessing_frame(self, observation):
        """
        0) atari frames: 210 x 160
        1) get image grayscale
        2) rescale image 110 x 84
        3) crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13: 110-13, :]
