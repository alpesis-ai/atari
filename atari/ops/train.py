import time
import threading

import gym
import tensorflow as tf

import settings
from core.nets.onestep import actor_learner_thread


def train(session, graph_ops, num_actions, saver):
    """
    Train a model.
    """

    # set up game environments (one per thread)
    envs = [gym.make(settings.GAME) for i in range(settings.N_THREADS)]
    summary_ops = _build_summaries()
    summary_op = summary_ops[-1]

    # initialize variables
    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(settings.SUMMARY_DIR + "/qlearning",
                                    session.graph)

    # initialize target network weights
    session.run(graph_ops["reset_target_network_params"])

    # start n_threads actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id,
                                                    envs[thread_id],
                                                    session,
                                                    graph_ops,
                                                    num_actions,
                                                    summary_ops,
                                                    saver)) \
                              for thread_id in range(settings.N_THREADS)]
    for t in actor_learner_threads:
        t.start()
        time.sleep(0.01)

    # show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if settings.SHOW_TRAINING:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > settings.SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(settings.T))
            last_summary_time = now

    for t in actor_learner_threads:
        t.join()




def _build_summaries():
    """
    Set up some episode summary ops to visualize on tensorboard
    """

    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Reward", episode_reward)

    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Qmax Value", episode_ave_max_q)

    logged_epsilon = tf.Variable(0.)
    tf.scalar_summary("Epsilon", logged_epsilon)

    # threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the grap)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op
