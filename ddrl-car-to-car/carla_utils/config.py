# HOST = "localhost"
HOST = "192.168.1.45"
PORT = 2_000
CLIENT_TIMEOUT = 10.0

DEFAULT_VEHICLE_MODEL = "vehicle.audi.tt"

DEFAULT_RGB_CAMERA_FOV = 110
DEFAULT_RGB_CAMERA_IM_WIDTH = 640
DEFAULT_RGB_CAMERA_IM_HEIGHT = 480
DEFAULT_RGB_CAMERA_HISTORY_LENGTH = 4


MIN_CITY_NUMBER = 1
MAX_CITY_NUMBER = 7
AVAILABLE_CITIES = [f"Town{a if a >= 10 else f'0{a}'}" for a in range(MIN_CITY_NUMBER, MAX_CITY_NUMBER+1)]

TOWNS_TURNS_SP = {
    "Town03": [
        122,
        123,
        127,
        128,
        136,
        154,
        156,
        166,
        173,
        175,
        176,
        187,
        188,
        194,
        195,
        196,
        197,
        210,
        211,
        218,
        219,
        229,
        231,
        232,
        233,
        243,
        244,
        245,
        246,
        247,
        248,
        252,
        263
    ],
    "Town01": [
        17,
        44,
        57,
        61,
        76,
        108,
        110,
        111,
        113,
        127,
        129,
        167,
        174,
        184,
        185,
        187,
        210,
        212,
        214,
        233,
        237,
        241
    ]
}
