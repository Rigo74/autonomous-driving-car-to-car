from models import *

INITIAL_EPSILON = 1  # initial value of epsilon
FINAL_EPSILON = 0.1  # 0.01  # final value of epsilon

# dipende da quanti step ci sono in un episodio,
# provereri a settarlo in modo da avere epsilon
# tra 0.3 e 0.4 a metà training i.e. dopo 500 episodi
# 0.3 < x^500 < 0.4
EPSILON_DECAY = 0.9993  # 0.97 0.95 0.9975 99975

EPISODES = 20_000  # 100 1000
SECONDS_PER_EPISODE = 10  # 40 10
STEPS_PER_EPISODE = 100
STEP_TIME_SECONDS = 0.7

AGGREGATE_STATS_EVERY_X_EPISODES = 100

MODEL_NAME = Cnn64x3.get_model_name()

REPLAY_MEMORY_SIZE = 100_000  # 50_000  # 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINI_BATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = 1  # MINI_BATCH_SIZE // 4
UPDATE_TARGET_EVERY_X_STEPS = 1_000  # 5

DISCOUNT = 0.99

# Custom Metrics
EPISODE_AVERAGE_REWARD = "Episode average reward"
EPSILON = "Epsilon"
AVERAGE_REWARD = f"Average reward of last {AGGREGATE_STATS_EVERY_X_EPISODES} episodes"
AVERAGE_LOSS = f"Average loss of last {AGGREGATE_STATS_EVERY_X_EPISODES} episodes"
EPISODE_AVERAGE_LOSS = "Episode average loss"
STEPS_PER_EPISODE_LABEL = "Steps done per episode"
