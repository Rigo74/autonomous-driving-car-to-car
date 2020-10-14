TURN = 0.1
IN_SPEED_LIMIT = 0.2
FORWARD_AT_INTERSECTION_ORANGE = 0.2 # non usato
FORWARD = 0.4
STOP_AT_INTERSECTION_ORANGE = 0.4 # non usato
STOP_AT_INTERSECTION_RED = 0.6
FORWARD_AT_INTERSECTION_GREEN = 0.6 # non usato
TARGET_REACHED = 1 # nel nostro caso non avverrà mai

OPPOSITE_TURN = -0.2 # prima giri in una direzione e subito dopo nell'altra
STOP_AT_INTERSECTION_GREEN = -0.3 # non usato
OVER_SPEED_LIMIT = -0.4
FORWARD_AT_INTERSECTION_RED = -1
WRONG_SIDE_ROAD = -0.95 # non possiamo saperla senza fare follie quindi per adesso è anche corsia sbagliata ma ne giusto senso di marcia
OFF_ROAD = -1
CRASH = -1
PEDESTRIAN_HIT = -1

###################################

MIN_REWARD = -3
