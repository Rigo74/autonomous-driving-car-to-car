TARGET_REACHED = 1  # nel nostro caso non avverrà mai
STOP_AT_INTERSECTION_RED = 0.6
CORRECT_SIDE_ROAD = 0 # 0.5
FORWARD = 0.05  # 0.4
TURN = 0.03  # 0.1
IN_SPEED_LIMIT = 0  # 0.1

OPPOSITE_TURN = -0.05  # -0.2 # prima giri in una direzione e subito dopo nell'altra
UNDER_MINIMUM_SPEED_LIMIT = 0 # -0.3
STOPPED_TOO_LONG = -0.3
# STOP_AT_INTERSECTION_GREEN = -0.3
OVER_SPEED_LIMIT = 0 # -0.4
WRONG_LANE = -0.7
WRONG_SIDE_ROAD = -0.95
FORWARD_AT_INTERSECTION_RED = -1
OFF_ROAD = -1
CRASH = -1
PEDESTRIAN_HIT = -1

###################################

MIN_REWARD = -300

# FORWARD_AT_INTERSECTION_GREEN = 0.6
# STOP_AT_INTERSECTION_ORANGE = 0.4
# FORWARD_AT_INTERSECTION_ORANGE = 0.2
