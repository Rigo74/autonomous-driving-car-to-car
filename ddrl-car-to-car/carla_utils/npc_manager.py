import carla
import random
import math

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor


class NPCManager(object):

    def __init__(self, client, world,
                 tm_port=8000,
                 number_of_vehicles_to_spawn=0,
                 number_of_pedestrians_to_spawn=0,
                 sync=False,
                 vehicles_filter='vehicle.*',
                 pedestrians_filter='walker.pedestrian.*',
                 percentage_pedestrians_running=0.0,  # how many pedestrians will run
                 percentage_pedestrians_crossing=0.0  # how many pedestrians will walk through the road
                 ):
        self.client = client
        self.world = world
        self.sync = sync

        self.vehicles_list = []
        walkers_list = []
        self.controllers_and_walkers_ids = []
        self.controllers_and_walkers_actors = []
        self.synchronous_master = False
        number_of_vehicles = number_of_vehicles_to_spawn
        number_of_pedestrians = number_of_pedestrians_to_spawn
        traffic_manager = client.get_trafficmanager(tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        if self.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                self.synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                self.synchronous_master = False

        blueprints = world.get_blueprint_library().filter(vehicles_filter)
        blueprints_walkers = world.get_blueprint_library().filter(pedestrians_filter)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # --------------
        # Spawn vehicles
        # --------------
        for i in range(0, math.ceil(number_of_vehicles/number_of_spawn_points)):
            random.shuffle(spawn_points)
            remaining_vehicles_to_spawn = number_of_vehicles - len(self.vehicles_list)
            self.vehicles_list.extend(
                self.spawn_vehicles(
                    number_of_vehicles=remaining_vehicles_to_spawn if remaining_vehicles_to_spawn <= number_of_spawn_points else number_of_spawn_points,
                    spawn_points=spawn_points,
                    blueprints=blueprints,
                    traffic_manager=traffic_manager
                )
            )

        # -------------
        # Spawn Walkers
        # -------------
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_pedestrians):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints_walkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if random.random() > percentage_pedestrians_running:
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if not results[i].error:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            self.controllers_and_walkers_ids.append(walkers_list[i]["con"])
            self.controllers_and_walkers_ids.append(walkers_list[i]["id"])
        self.controllers_and_walkers_actors = world.get_actors(self.controllers_and_walkers_ids)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if sync and self.synchronous_master:
            world.tick()
        else:
            world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)
        for i in range(0, len(self.controllers_and_walkers_ids), 2):
            # start walker
            self.controllers_and_walkers_actors[i].start()
            # set walk to random point
            self.controllers_and_walkers_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            self.controllers_and_walkers_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

    def destroy(self):
        if self.sync and self.synchronous_master:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.vehicles_list.clear()

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.controllers_and_walkers_actors), 2):
            self.controllers_and_walkers_actors[i].stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers_and_walkers_ids])

        self.controllers_and_walkers_ids.clear()
        self.controllers_and_walkers_actors = None

    def spawn_vehicles(self, number_of_vehicles, spawn_points, blueprints, traffic_manager):
        batch = []
        for i in range(0, number_of_vehicles):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(
                SpawnActor(blueprint, spawn_points[i]).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        return [x.actor_id for x in self.client.apply_batch_sync(batch, self.synchronous_master) if not x.error]
