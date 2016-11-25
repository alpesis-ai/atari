import tflearn
import tensorflow as tf


def build_dqn(num_actions, action_repeat):
    """
    Build a DQN.
    """

    # input shape: [batch, channel, height, width]
    inputs = tf.placeholder(tf.float32, [None, action_repeat, 84, 84])
    # input shape changes to be: [batch, height, width, channel]
    net = tf.transpose(inputs, [0, 2, 3, 1])
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    q_values = tflearn.fully_connected(net, num_actions)

    return inputs, q_values
