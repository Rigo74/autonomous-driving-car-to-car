import carla

CLIENT_ADDRESS = "localhost"
CLIENT_PORT = 2000
TOWN = "Town02"

client = carla.Client(CLIENT_ADDRESS, CLIENT_PORT)
world = client.load_world(TOWN)

weather = carla.WeatherParameters.ClearNoon

world.set_weather(weather)
