import argparse
import time
import msgpack
from enum import Enum, auto
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from shapely.geometry import LineString, Polygon, Point
from sklearn.neighbors import KDTree

from planning_utils import a_star, heuristic, create_grid
from planning_utils import create_graph_and_edges, find_start_goal
from planning_utils import a_star_graph, condense_waypoints, extract_polygons, construct_road_map
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    # self.plan_path()
                    self.plan_path_graph()
                    # self.prob_road_map()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1],
                          self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def read_home_lat_lon(self, file_path):
        lat, lon = None, None
        with open(file_path) as f:
            first_line = f.readline()
            lat_lon_str = first_line.split(",")
            lat_lon_str = list(map(str.strip, lat_lon_str))
            lat = lat_lon_str[0].split(" ")[1]
            lon = lat_lon_str[1].split(" ")[1]

        return float(lat), float(lon)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 8

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = self.read_home_lat_lon('colliders.csv')

        # TODO: set home position to (lon0, lat0, 0)
        print(f'Setting home to (lon0, lat0, 0): ({lon0, lat0, 0})')
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        global_position = self.global_position

        # TODO: convert to current local position using global_to_local()
        local_home = global_to_local(global_position, self.global_home)
        print(f'Home coordinates in local form: {local_home}')

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        grid_start = (int(self.local_position[0] - north_offset),
                      int(self.local_position[1] - east_offset))

        # Set goal as some arbitrary position on the grid
        # grid_goal = (grid_start[0] + 10, grid_start[1] + 10)
        # grid_goal = (430, 350)  # this is a good test goal state
        # TODO: adapt to set goal as latitude / longitude position and convert
        # (430, 350) corresponds to (lat, lon) == (37.793510, -122.398525)
        goal_lon_lat = [-122.398525, 37.793510, 0]
        # goal_lon_lat = [-122.398250, 37.793875, 0]
        # goal_lon_lat = [-122.398275, 37.796875, 0]
        # goal_lon_lat = [-122.399970, 37.795947, 0]  # complicted path
        # goal_lon_lat = [-122.393284, 37.790545, 0]
        # goal_lon_lat = [-122.401173, 37.795514, 0]  # (point1)
        local_position_goal = global_to_local(goal_lon_lat, self.global_home)
        grid_goal = (int(local_position_goal[0] - north_offset), 
                     int(local_position_goal[1] - east_offset))
        # print(grid_goal)

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()
        
    def plan_path_graph(self):
        """
        Construct a configuration space using a graph representation, set destination GPS position, find a path from start (current) position to destination, and minimize and set waypoints.
        """
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = self.read_home_lat_lon('colliders.csv')

        # TODO: set home position to (lon0, lat0, 0)
        print(f'Setting home to (lon0, lat0, 0): ({lon0, lat0, 0})')
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        global_position = self.global_position

        # TODO: convert to current local position using global_to_local()
        local_home = global_to_local(global_position, self.global_home)
        print(f'Home coordinates in local form: {local_home}')

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a graph for a particular altitude and safety margin around obstacles
        grid, G, north_offset, east_offset = create_graph_and_edges(
            data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        # TODO: convert start position to current position rather than map center
        grid_start = (int(self.local_position[0] - north_offset),
                      int(self.local_position[1] - east_offset))
        # goal_lon_lat = [-122.398525, 37.793510, 0]
        # goal_lon_lat = [-122.398250, 37.793875, 0]
        # goal_lon_lat = [-122.398275, 37.796875, 0]  # (point3) reachable if SAFETY_DISTANCE = 5
        # goal_lon_lat = [-122.399970, 37.795947, 0]  # reachable if SAFETY_DISTANCE = 5
        # goal_lon_lat = [-122.393284, 37.790545, 0]  # (point2)
        goal_lon_lat = [-122.401173, 37.795514, 0]  # (point1)
        # goal_lon_lat = [-122.393805, 37.795825, 0] # not reachable if TARGET_ALTITUDE = 5
        # (point1) -> (point2) -> (point3) this sequence results in some
        # trees getting in the way of the drone. This is a good example
        # of imprecise maps and we may have to use receding horizon 
        # planning which utilizes a 3D state space for a limited region
        # around the drone.
        grid_goal = global_to_local(goal_lon_lat, self.global_home)
        grid_goal = (int(grid_goal[0] - north_offset), 
                     int(grid_goal[1] - east_offset))

        # Find closest node on the Vornoi graph
        g_start, g_goal = find_start_goal(G, grid_start, grid_goal)
        print("Start and Goal location:", grid_start, grid_goal)
        print("Start and Goal location on Vornoi graph:", g_start, g_goal)
        path, _ = a_star_graph(G, heuristic, g_start, g_goal)
        # Add the final goal state (instead of the approximated 
        # goal on the graph)
        path.append(grid_goal)
        # print(path)
        # Reduce waypoints
        path = condense_waypoints(grid, path)
        # print(path)
        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def prob_road_map(self):
        """Probablistic Road Map implementation to create configuration space, set start and goal positions, find path from start to goal, and condense and set waypoints for navigation.
        """
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0, lon0 = self.read_home_lat_lon('colliders.csv')

        # TODO: set home position to (lon0, lat0, 0)
        print(f'Setting home to (lon0, lat0, 0): ({lon0, lat0, 0})')
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        global_position = self.global_position

        # TODO: convert to current local position using global_to_local()
        local_home = global_to_local(global_position, self.global_home)
        print(f'Home coordinates in local form: {local_home}')

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        print(f'Constructing Probablistic Road Map...')
        st = time.time()
        grid, G, north_offset, east_offset = construct_road_map(
            data, TARGET_ALTITUDE, SAFETY_DISTANCE, 500, 4)
        time_taken = time.time() - st
        print(f'construct_road_map() took: {time_taken} seconds')
        # TODO: convert start position to current position rather than map center
        grid_start = (int(self.local_position[0] - north_offset),
                      int(self.local_position[1] - east_offset))
        # goal_lon_lat = [-122.398525, 37.793510, 0]
        # goal_lon_lat = [-122.398250, 37.793875, 0]
        # goal_lon_lat = [-122.398275, 37.796875, 0]  # (point3) reachable if SAFETY_DISTANCE = 5
        # goal_lon_lat = [-122.399970, 37.795947, 0]  # reachable if SAFETY_DISTANCE = 5
        # goal_lon_lat = [-122.393284, 37.790545, 0]  # (point2)
        goal_lon_lat = [-122.401173, 37.795514, 0]  # (point1)
        # goal_lon_lat = [-122.393805, 37.795825, 0] # not reachable if TARGET_ALTITUDE = 5
        # (point1) -> (point2) -> (point3) this sequence results in some
        # trees getting in the way of the drone. This is a good example
        # of imprecise mapping and we may have to use receding horizon 
        # planning which utilizes a 3D state space for a limited region
        # around the drone.
        grid_goal = global_to_local(goal_lon_lat, self.global_home)
        grid_goal = (int(grid_goal[0] - north_offset), 
                     int(grid_goal[1] - east_offset))

        # Find closest node on probablistic road map graph
        g_start, g_goal = find_start_goal(G, grid_start, grid_goal)
        print("Start and Goal location:", grid_start, grid_goal)
        print("Start and Goal location on graph:", g_start, g_goal)
        path, _ = a_star_graph(G, heuristic, g_start, g_goal)
        # Add the final goal state (instead of the approximated 
        # goal on the graph)
        path.append(grid_goal)
        print(path)
        # Reduce waypoints
        path = condense_waypoints(grid, path)
        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=600)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
