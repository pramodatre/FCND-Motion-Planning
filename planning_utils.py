from enum import Enum
from queue import PriorityQueue
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from bresenham import bresenham
import numpy.linalg as LA
from shapely.geometry import LineString, Polygon, Point
from sklearn.neighbors import KDTree
from random import sample 
import time
import matplotlib.pyplot as plt


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


def heuristic(position, goal_position):
    """[summary]

    Args:
        position ([type]): [description]
        goal_position ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def create_graph_and_edges(data, drone_altitude, safety_distance):
    """Returns a grid representation of a 2D configuration space along with Voronoi graph edges given obstacle data and the drone's altitude.

    Args:
        data (pandas.DataFrame): containing obstacle centers and extensions in north and east directions
        drone_altitude (float): desired altitude of flight
        safety_distance (float): desired safety distance from obstacles

    Returns:
        grid: 2D occupancy map at 1m-by-1m resolution
        G: Graph with nodes and edges of the configuration space
        north_min: offset for north direction
        east_min: offset for east direction
    """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])

    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)
    print(len(graph.ridge_vertices))
    # voronoi_plot_2d(graph)
    # TODO: check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        #x1, y1 = int(p1[0]), int(p1[1])
        #x2, y2 = int(p2[0]), int(p2[1])
        #print(x1, y1)
        #print(x2, y2)
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        # print(cells)
        # print(grid.shape)
        hit = False
        for c in cells:
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            if grid[c[0], c[1]] == 1:
                hit = True
                break

        if not hit:
            p1 = (int(p1[0]), int(p1[1]))
            p2 = (int(p2[0]), int(p2[1]))
            edges.append((p1, p2))

    G = nx.Graph()

    for e in edges:
        p1 = tuple(e[0])
        p2 = tuple(e[1])
        dist = LA.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)

    return grid, G, int(north_min), int(east_min)


def euclidean_dist(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def find_start_goal(graph, start, goal):
    """Find the closest node in the graph to the specified start and goal location specifed as argument. This mapping is necessary since the start and goal positions are not bound by locations in the configuration space.

    Args:
        graph (nx.Graph): containing nodes and edges
        start (tuple): containing start north and east coordinate
        goal (tuple): containing goal north and east coordinate

    Returns:
        near_start (tuple): containing closest node in the graph to the specified start location
        near_goal (tuple): containing closest node in the graph to the specified goal location
    """
    # TODO: find start and goal in the graph
    near_start = None
    near_goal = None

    near_start_list = []
    near_goal_list = []

    first_iter = True
    min_dist_start = None
    min_dist_goal = None
    for n in graph:
        # print(n)
        start_dist = euclidean_dist(np.array(n), np.array(start))
        goal_dist = euclidean_dist(np.array(n), np.array(goal))
        if first_iter:
            min_dist_start = start_dist
            min_dist_goal = goal_dist
            near_start = n
            near_goal = n
            first_iter = False
        else:
            if start_dist < min_dist_start:
                #print(f'near_start: {n}')
                near_start_list.append(n)
                min_dist_start = start_dist
                near_start = n

            if goal_dist < min_dist_goal:
                #print(f'near_goal: {n}')
                near_goal_list.append(n)
                min_dist_goal = goal_dist
                near_goal = n

    return near_start, near_goal


# Modified A* that works with graph
def a_star_graph(G, h, start, goal):
    """Perform A* search from start to goal using the specified heuristic function over the graph G.

    Args:
        G (nx.Graph): containing nodes and edges of the configuration space
        h (function): heuristic function, e.g., euclidean distance
        start (node): start node of nx.Graph
        goal ([type]): goal node of nx.Graph

    Returns:
        path (list): containing coordinates of waypoints
        cost (float): path cost
    """
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]

        if current_node == goal:       
            print('Found a path.')
            found = True
            break
        else:
            # For a graph, next possible action is the
            # node that is closest to the current node
            for n_node in G.neighbors(current_node):
                edge_data = G.get_edge_data(current_node, n_node)

                # get the tuple representation
                # da = action.delta
                # next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + edge_data['weight']
                queue_cost = branch_cost + h(n_node, goal)

                if n_node not in visited:                
                    visited.add(n_node)               
                    branch[n_node] = (branch_cost, current_node, n_node)
                    queue.put((queue_cost, n_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


def condense_waypoints(grid, path):
    """
    Apply Bresenham over all points in the path and retain
    minimal waypoints. Will retain fisrt and last waypoint.

    Args:
        grid (numpy.ndarray): representing occupancy of cells
             where 1 represents occupied and 0 represents 
             not-occupied
        path (list): waypoints possibly containg redundant points

    Returns:
        (list): containing munimal points from start to end point
                in path
    """
    sampled_waypoints = []
    last_p2 = None
    for p in path:
        if not sampled_waypoints:
            sampled_waypoints.append((int(p[0]), int(p[1])))
            last_p2 = p
        else:
            p1 = sampled_waypoints[-1]
            cells = list(bresenham(int(p1[0]), int(p1[1]), 
                                   int(p[0]), int(p[1])))
            hit = False
            for c in cells:
                if grid[c[0], c[1]] == 1:
                    hit = True

            if not hit:
                last_p2 = p
            else:
                sampled_waypoints.append((int(last_p2[0]), int(last_p2[1])))
    # Ensure last point in the path is in the condensed path
    p_last = path[-1]
    if not (int(p_last[0]), int(p_last[1])) in sampled_waypoints:
        sampled_waypoints.append((int(p_last[0]), int(p_last[1])))

    return sampled_waypoints

# Create polygons for obstacles using x and y coordinates
# and store z coordinate with the polygon
def extract_polygons(data, apply_offset=True):

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    alt_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))
    alt_size = int(alt_max)

    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # TODO: Extract the 4 corners of the obstacle
        # 
        # NOTE: The order of the points matters since
        # `shapely` draws the sequentially from point to point.
        #
        # If the area of the polygon is 0 you've likely got a weird
        # order.
        # corners = [None, None, None, None]
        if apply_offset:
            corners = [
                (north - d_north - north_min, east - d_east - east_min),
                (north - d_north - north_min, east + d_east - east_min),
                (north + d_north - north_min, east + d_east - east_min),
                (north + d_north - north_min, east - d_east - east_min)

            ]
        else:
            corners = [
                (north - d_north, east - d_east),
                (north - d_north, east + d_east),
                (north + d_north, east + d_east),
                (north + d_north, east - d_east)

            ]
        # TODO: Compute the height of the polygon
        height = alt + d_alt

        # TODO: Once you've defined corners, define polygons
        p = Polygon(corners)
        polygons.append((p, height))

    return polygons


def collides_kdtree(polygons, tree, point):
    # TODO: Determine whether the point collides
    # with any obstacles.
    x, y, z = point
    p = Point(x, y)
    p_array = np.array([x, y])
    # Get closest polygon
    idx = tree.query([p_array], 
                     k=1, return_distance=False)[0]
    obstacle = polygons[idx[0]]
    o_poly = obstacle[0]
    o_height = obstacle[1]
    if o_poly.contains(p) and z < o_height:
        return True

    return False


def can_connect(polygons, p1, p2):
    l = LineString([p1, p2])
    for poly, height in polygons:
        if l.crosses(poly):
            return False
    return True


def can_connect_fast(polygons, tree_poly, p1, p2):
    """
    Faster implementation of can_connect() method using random sampling of points on the line between p1 and p2 and KDTree to query for polygons that are close to the points

    Args:
        polygons (tuple): containing (polygon, height) for all obstacles
        tree_poly (KDTree): indexed using ploygon vertices
        p1 (tuple): containing north, east coordinate of first point
        p2 (tuple): containing north, east coordinate of first point

    Returns:
        (bool): True of p1 and p2 can be connected. False otherwise.
    """
    # Find closest poly for p1 and p2 and check if line 
    # intersects with one of them
    l = LineString([p1, p2])
    # Find closeset polygon to the points
    idx = tree_poly.query([np.array(p1)], 
                          k=1, return_distance=False)[0]
    poly1 = polygons[idx[0]][0]
    if l.crosses(poly1):
        return False
    # Sample points on the line
    # for i in [0.25, 0.5, 0.75]:
    for i in np.arange(0.1, 0.9, 0.1):
        p = l.interpolate(i, normalized=True)
        # print(p)
        idx = tree_poly.query([np.array(p)], 
                              k=1, return_distance=False)[0]
        poly2 = polygons[idx[0]][0]
        # If line crosses the any polygon, we cannot connect
        if l.crosses(poly2):
            return False

    return True


def construct_road_map(data, drone_altitude, safety_distance, num_nodes, neighbors):
    """Probablistic Road Map construction by randomly sampling points, check if points and edges between them are feasible using occupancy, initialize start and goal nodes, find and return path betwen start and goal nodes.

    Args:
        data (pandas.DataFrame): containing obstacle centers and extensions in north and east directions
        drone_altitude (float): desired altitude of flight
        safety_distance (float): desired safety distance from obstacles
        num_nodes (int): to be sampled
        neighbors (int): number of neighbors to check for possible connections

    Returns:
        grid: 2D occupancy map at 1m-by-1m resolution
        g: Graph with nodes and edges of the configuration space
        north_min: offset for north direction
        east_min: offset for east direction
    """
    grid, north_min, east_min = create_grid(
        data, drone_altitude, safety_distance)

    print("Completed grid construction...")
    polygons = extract_polygons(data, apply_offset=False)
    poly_bounds = []
    for poly, height in polygons:
        poly_bounds.append(np.array(list(poly.centroid.coords))[0])
    print("Completed polygon creation for obstacles...")
    tree_poly = KDTree(np.array(poly_bounds))
    print("Completed KDTree cretation for polygons...")
    g = nx.Graph()
    # Randomly sample points
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    zmin = 0
    # Limit the z axis for the visualization
    zmax = 10

    # Randomly sample nodes to be added
    num_samples = 50
    to_keep = []
    while len(to_keep) < num_nodes:
        xvals = np.random.randint(xmin, xmax, num_samples)
        yvals = np.random.randint(ymin, ymax, num_samples)
        zvals = np.random.randint(zmin, zmax, num_samples)
        samples = list(zip(xvals, yvals, zvals))
        # Check for collission free nodes
        for point in samples:
            if not collides_kdtree(polygons, tree_poly, point):
                to_keep.append(point)
    print("Completed node selection...")
    # randomly sample nodes from to_keep, find neighbors 
    # and add edges when fesible
    edges = {}
    while len(g.edges) < num_nodes:
        nodes_sampled = sample(to_keep, num_samples)
        # print(nodes_sampled)
        nodes = []
        for point in nodes_sampled:
            nodes.append([point[0], point[1]])
        tree = KDTree(np.array(nodes))
        # print("KDTree created for the sampled nodes...")
        for n in nodes:
            # print("Querying the KKDTree...")
            idx = tree.query([np.array(n)], 
                             k=neighbors, return_distance=False)[0]
            # print("Got back response...")
            # check if node n is connectable to 
            # the nodes that idx refers to
            for i in idx:
                n_i = np.array(nodes[i])
                if tuple(n) == tuple(n_i):
                    continue
                if tuple(n) in edges:
                    if edges[tuple(n)] == tuple(n_i):
                        continue
                if tuple(n_i) in edges:
                    if edges[tuple(n_i)] == tuple(n):
                        continue
                # print("Checking can_connect() ...")
                # Too slow on the simulator!
                if can_connect_fast(polygons, tree_poly, Point(n), Point(n_i)):
                    g.add_edge(tuple(n), tuple(n_i), weight=1)
                    edges[tuple(n)] = tuple(n_i)
                    edges[tuple(n_i)] = tuple(n)
                    # print("can be connected.")

                # cells = list(bresenham(int(n[0]), int(n[1]), int(n_i[0]), int(n_i[1])))
                # # print(cells)
                # # print(grid.shape)
                # hit = False
                # for c in cells:
                #     if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                #         hit = True
                #         break
                #     if grid[c[0], c[1]] == 1:
                #         hit = True
                #         break

                # if not hit:
                #     g.add_edge(tuple(n), tuple(n_i), weight=1)
        # print("Completed adding feasible edges to the batch...")
        # print("Number of edges", len(g.edges))
    # Visualize graph
    # visualize_prob_road_map(data, grid, g)
    print("Completed all edge additions.")

    return grid, g, int(north_min), int(east_min)

def visualize_prob_road_map(data, grid, g):
    fig = plt.figure()
    plt.imshow(grid, cmap='Greys', origin='lower')
    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])
    # If you have a graph called "g" these plots should work
    # Draw edges
    for (n1, n2) in g.edges:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black', alpha=0.5)
    # Draw all nodes connected or not in blue
    # for n1 in nodes:
    #     plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
    # Draw connected nodes in red
    for n1 in g.nodes:
        plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    plt.xlabel('NORTH')
    plt.ylabel('EAST')
    plt.show()


if __name__ == '__main__':
    # Read in obstacle map
    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
    TARGET_ALTITUDE = 5
    SAFETY_DISTANCE = 5
    st = time.time()
    construct_road_map(data, TARGET_ALTITUDE, SAFETY_DISTANCE, 500, 4)
    time_taken = time.time() - st
    print(f'construct_road_map() took: {time_taken} seconds')
