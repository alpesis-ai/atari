# ------------------------------------------------------
# Game settings

# game: 'Breakout-v0', 'Pong-v0', SpaceInvaders-v0', ...
GAME = 'MsPacman-v0'

# change that value to test instead of train
TESTING = False

# learning threads
N_THREADS = 8

# ------------------------------------------------------
# Training Parameters

# consecutive screen frames when performing training
ACTION_REPEAT = 4
# Learning rate
LEARNING_RATE = 0.001


# Max Training steps
TMAX = 80000000
# Current training step
T = 0

# Async gradient update frequency of each learning thread
I_ASYNCUPDATE = 5
# Timestep to reset the target network
I_TARGET = 40000

# reward discount rate
GAMMA = 0.99
# number of timesteps to anneal epsilon
ANNEAL_EPSILON_TIMESTEPS = 400000

# directory for storing tensorboard summaries
SUMMARY_DIR = '/tmp/tflearn_logs/'

# ------------------------------------------------------
# Test Parameters

# number of episodes to run gym evaluation
NUM_EVAL_EPISODES = 100

# test model path
# TEST_MODEL_PATH = '/path/to/qlearning.tflearn.ckpt'
TEST_MODEL_PATH = './qlearning.ckpt'

# ------------------------------------------------------
# Utils Parameters

# display or not gym environment screens
SHOW_TRAINING = True

# directory for storing tensorboard summaries
SUMMARY_DIR = '/tmp/tflearn_logs/'
SUMMARY_INTERVAL = 100
CHECKPOINT_PATH = './outputs/models/qlearning.tflearn.ckpt'
CHECKPOINT_INTERVAL = 2000
