import gym
import numpy as np

import settings
from core.env.atari import AtariEnvWrapper


def evaluate(session, graph_ops, saver):
    """
    Evaluate a model.
    """
    saver.restore(session, settings.TEST_MODEL_PATH)
    print("Restored model weights from ", settings.TEST_MODEL_PATH)

    monitor_env = gym.make(settings.GAME)
    monitor_env.monitor.start("qlearning/eval")

    # unpack graph ops
    s = graph_ops['s']
    q_values = graph_ops['q_values']

    # wrap env with AtariEnvWrapper helper class
    env = AtariEnvWrapper(gym_env=monitor_env,
                          action_repeat=settings.ACTION_REPEAT)

    for i_episode in xrange(settings.NUM_EVAL_EPISODES):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)
    monitor_env.monitor.close()
