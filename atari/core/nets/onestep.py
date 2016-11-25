import time
import random

import numpy as np

import settings
from core.env.atari import AtariEnvWrapper


def actor_learner_thread(thread_id,
                         env,
                         session,
                         graph_ops,
                         num_actions,
                         summary_ops,
                         saver):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning,
    as specified in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """

    # global settings.T
    # global settings.TMAX

    # unpack graph ops
    s = graph_ops['s']
    q_values = graph_ops['q_values']
    st = graph_ops['st']
    target_q_values = graph_ops['target_q_values']
    reset_target_network_params = graph_ops['reset_target_network_params']
    a = graph_ops['a']
    y = graph_ops['y']
    grad_update = graph_ops['grad_update']

    summary_placeholders, assign_ops, summary_op = summary_ops

    # wrap env with AtariEnviornment helper class
    env = AtariEnvWrapper(gym_env=env,
                          action_repeat=settings.ACTION_REPEAT)

    # initialize network gradients
    s_batch = []
    a_batch = []
    y_batch = []
    
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0
    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(3*thread_id)
    t = 0
    while settings.T < settings.TMAX:
        # get initial game observation
        s_t = env.get_initial_state()
        terminal = False

        # set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        while True:
            # forward the deep q network, get Q(s, a) values
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            # choose next action based on e-greedy policy
            a_t = np.zeros([num_actions])
            if random.random() <= epsilon:
                action_index = random.randrange(num_actions)
            else:
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / settings.ANNEAL_EPSILON_TIMESTEPS

            # gym excecutes action in game environment on behalf of action-learner
            s_t1, r_t, terminal, info = env.step(action_index)

            # accumulate gradients
            readout_j1 = target_q_values.eval(session=session, feed_dict={st: [s_t1]})
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + settings.GAMMA * np.max(readout_j1))

            a_batch.append(a_t)
            s_batch.append(s_t)

            # update the state and counters
            s_t = s_t1
            settings.T += 1
            t += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(readout_t)

            # optionally update target network
            if settings.T % settings.I_TARGET == 0:
                session.run(reset_target_network_params)

            # optionally update online network
            if t % settings.I_ASYNCUPDATE == 0 or terminal:
                if s_batch:
                    session.run(grad_update, feed_dict={y: y_batch,
                                                        a: a_batch,
                                                        s: s_batch})

                # clear gradients
                s_batch = []
                a_batch = []
                y_batch = []

            # save model progress
            if t % settings.CHECKPOINT_INTERVAL == 0:
                saver.save(session, "qlearning.ckpt", global_step=t)

            # print end of episode stats
            if terminal:
                stats = [ep_reward, episode_ave_max_q / float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i],
                                {summary_placeholders[i]: float(stats[i])})

                print("| Thread %.2i" % int(thread_id), "| Step", t,
                      "| Reward: %.2i" % int(ep_reward), " Qmax: %.4f" %
                      (episode_ave_max_q / float(ep_t)),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(settings.ANNEAL_EPSILON_TIMESTEPS)))
                break
            

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of 
        http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]
