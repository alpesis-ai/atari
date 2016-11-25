import tflearn
import tensorflow as tf

import settings
from core.nets import dqn


def build_graph(num_actions):
    """
    """

    # create shared deep q network
    s, q_network = dqn.build_dqn(num_actions=num_actions,
                                 action_repeat=settings.ACTION_REPEAT)
    network_params = tf.trainable_variables()
    q_values = q_network

    # create shared target network
    st, target_q_network = dqn.build_dqn(num_actions=num_actions,
                                         action_repeat=settings.ACTION_REPEAT)
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.mul(q_values, a), reduction_indices=1)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(settings.LEARNING_RATE)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s": s,
                 "q_values": q_values,
                 "st": st,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update}

    return graph_ops


