# -*- coding: utf-8 -*-
"""
Teaching a machine to play an Atari game (Pacman by default) by implementing
a 1-step Q-learning with TFLearn, TensorFlow and OpenAI gym environment. The
algorithm is described in "Asynchronous Methods for Deep Reinforcement Learning"
paper. OpenAI's gym environment is used here for providing the Atari game
environment for handling games logic and states. This example is originally
adapted from Corey Lynch's repo (url below).

Requirements:
    - gym environment (pip install gym)
    - gym Atari environment (pip install gym[atari])

References:
    - Asynchronous Methods for Deep Reinforcement Learning. Mnih et al, 2015.

Links:
    - Paper: http://arxiv.org/pdf/1602.01783v1.pdf
    - OpenAI's gym: https://gym.openai.com/
    - Original Repo: https://github.com/coreylynch/async-rl

"""

import gym
import tensorflow as tf

import settings
from graphs import graphs
from ops.train import train
from ops.evaluate import evaluate


def main(_):

    with tf.Session() as session:
        num_actions = get_num_actions()
        graph_ops = graphs.build_graph(num_actions)
        saver = tf.train.Saver(max_to_keep=5)

        if settings.TESTING:
            pass
            evaluate(session, graph_ops, saver)
        else:
            train(session, graph_ops, num_actions, saver)


def get_num_actions():
    """
    Returns the number of possible actions for the given atari game.
    """
    # figure out number of actions from gym env
    env = gym.make(settings.GAME)
    num_actions = env.action_space.n
    return num_actions


if __name__ == '__main__':

    tf.app.run()
