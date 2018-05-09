#!/usr/bin/env python

import sys
import math

import networkx as nx

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import quadtree.QuadTree as QT
import quadtree.Util as util
import quadtree.globals as glob

import numpy as np
import random

import cPickle

import argparse
 
  

def generate_road_network(limit=800, size=1000):
    glob.init()
    
    # 2 D plane
    # size=1000 # if aquares are not fully forming increase plane size
    X=util.getPlane(size)

    mins = (0.0, 0.0)
    maxs = (size-1.0, size-1.0)

    q = QT.QuadTree(X, mins, maxs, 0, 0)

    q.add_square()

    # for high density choose ones counter depth with highest number of squares randomly
    while(True):
     	node=random.randrange(max(glob.nodeIndex))
    	if len(glob.nodeIndex)>limit: # limit network generation by number of nodes
    		break

    	util.add_square_at(q,node)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, size-1.0)
    ax.set_ylim(0, size-1.0)

    # for each depth generate squares
    # print "generating squares..."
    for d in range(0,len(glob.nodeIndex)):
        q.draw_rectangle(ax, depth=d)

    id2rank = {}

    pos = {}


    nodeCount=0
    edgeCount=0 #directed edge count
    for point in glob.edges:
        if point in glob.coord_id:
            # fn.write(str(glob.coord_id[point])+","+str(point.x)+","+str(point.y)+"\n")
            # print nodeCount, point, glob.coord_id[point]
            id2rank[glob.coord_id[point]] = nodeCount
            pos[nodeCount] = (point.x, point.y)
            nodeCount += 1
        
    # print pos
        
    G = nx.Graph()
    G.add_nodes_from( range(nodeCount) )

    for point in glob.edges:
    	for edge in glob.edges[point]:
            # fe.write(str(glob.coord_id[point])+","+str(glob.coord_id[edge])+"\n")
            edgeCount=edgeCount+1
            x = id2rank[glob.coord_id[point]]
            y = id2rank[glob.coord_id[edge]]
            d = point.distTo(edge)
            G.add_edge( x, y, distance=d )
            # print id2rank[glob.coord_id[point]], id2rank[glob.coord_id[edge]], point.distTo(edge)
    
    return G, pos


def capital(G, constraint=lambda x: True, ties=lambda x,y:x):
    c = None
    d = sys.maxint
    d2 = sys.maxint
    for node in list(G.nodes):
        if len(G[node]) == 4 and constraint(node):
            furthest = max([G.edges[node,neighbor]['distance'] for neighbor in G[node]])
            if furthest < d or (furthest == d and ties(c,node) == node):
                d = furthest
                c = node
    return c
    
def find_closest(pos, x, y, nodes=None):
    if nodes is None:
        nodes = list(pos.keys())
    closest = nodes[0]
    mind = distance(pos[closest], (x,y))
    for node in nodes[1:]:
        d = distance(pos[node], (x,y))
        if d < mind:
            mind = d
            closest = node
    return closest 
    
def distance((x,y), (a,b)):
    return math.sqrt((x-a)*(x-a) + (y-b)*(y-b))

def closer_to_corner( (x,y), (a,b), size ):
    return min(((distance((x,y),(0,0))), (distance((x,y),(0,size))), (distance((x,y),(size,0))), (distance((x,y),(size,size))))) < min(((distance((a,b),(0,0))), (distance((a,b),(0,size))), (distance((a,b),(size,0))), (distance((a,b),(size,size)))))
       
def sort_around(x,y, pos):
    return [n for d,n in sorted([(distance(pos[node], (x,y)), node) for node in pos.keys])]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def burn(fire, burnt, visible, wind=(0,0), intensity=.25, step=1, size=1000, foyer=(0,0), direction=(0,0)):
    foyer_x,foyer_y = foyer
    d_x,d_y = direction
    d_x *= len(burnt)
    d_y *= len(burnt)
    propagation = set([])
    estinguished = set([])
    for x,y in fire:
        no_propagation = len(propagation)
        for dx in range(-1*step,step+1,step):
            for dy in range(-1*step,step+1,step):
                if (dx != 0 or dy != 0) and (x+dx, y+dy) not in burnt:
                    A = angle_between(wind, (dx,dy))
                    p = intensity * ((math.pi-A) * (math.pi-A)) / (math.pi * math.pi)
                    
                    rd = random.uniform(0,1)
                    
                    # print x,y, '[%i/%i]'%(dx,dy), rd, '<>', p
                    if rd < p:
                        propagation.add( (x+dx, y+dy) )
                        d_x += (x+dx - foyer_x)
                        d_y += (y+dy - foyer_y)
                        if x+dx+step >= 0 and x+dx < size and y+dy+step >= 0 and y+dy < size :
                            visible.add( (x+dx, y+dy) )
                        
        if len(propagation) == no_propagation:
            if random.uniform(0,1) < (intensity*intensity):
                estinguished.add( (x,y) )
    
        fire = ((fire | propagation) - estinguished)
        burnt = (burnt | fire)
        
        
 
    return fire, burnt, visible, (d_x/float(len(burnt)), d_y/float(len(burnt)))
    
# def intersect( A, B, C, D ):
#     # if max(x1,x2) >= min(x3,x4) and min(x1,x2) <= max(x3,x4) and max(y1,y2) >= min(y3,y4) and min(y1,y2) <= max(y3,y4):
#     return counterclockwise(A,C,D) != counterclockwise(B,C,D) and counterclockwise(A,B,C) != counterclockwise(A,B,D)

def intersect( xmin, xmax, ymin, ymax, (x1,y1), (x2,y2) ): # the edge is either horizontal or vertical
    return min(x1,x2) <= xmax and max(x1,x2) >= xmin and min(y1,y2) <= ymax and max(y1,y2) >= ymin
    
def box_hull( points ):
    Xs = [x for x,y in points]
    Ys = [y for x,y in points]
    return (min(Xs), max(Xs)), (min(Ys), max(Ys))   

def counterclockwise((x1,y1), (x2,y2), (x3,y3)):
    return (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
         
def convex_hull( points ):
    x0,y0 = points[0]
    for i in range(1,len(points)):
        xi,yi = points[i]
        if yi < y0:
            x0,y0 = points[i]
            points[i] = points[0]
            points[0] = x0,y0      

    x0,y0 = points[0]
    S = sorted([(angle_between((1,0), (x-x0, y-y0)), (x,y)) for x,y in points[1:]])
    
    points = [S[-1][1],(x0,y0)]
    for A,p in S:
        points.append(p)

    # M will denote the number of points on the convex hull.
    M = 1
    N = len(points)
    for i in range(2,N):
        # Find next valid point on convex hull.
        while counterclockwise(points[M-1], points[M], points[i]) <= 0:
            if M > 1:
                M -= 1
                continue
            # All points are collinear
            elif i == N:
                break
            else:
                i += 1
        # Update M and swap points[i] to the correct place.
        M += 1
        x,y = points[M]
        points[M] = points[i]
        points[i] = x,y

    return points[:M]          


def treeify(route, tree):
    treeified = []
    for v in route:
        for path in tree:
            for i in range(len(path)):
                u = path[i]
                if v == u:
                    treeified.extend(path[i:])
                    return treeified
        treeified.append(v)
    return treeified
                    

def write_evacuation_plan(G, threatened_nodes, safe_zone, num_evacuations, time_factor, filename):
    # num_evacuations = min(num_evacuations,len(threatened_nodes))            
    if num_evacuations > len(threatened_nodes):         
        print 'bad instance: not enough fire'
        return safe_nodes[random.randint(0, len(safe_nodes))], []
    if len(safe_nodes) == 0:
        print 'bad instance: too much fire'
        return None, []
    
    
    for i in range(num_evacuations):
        j = random.randint(i, len(threatened_nodes)-1)
        v = threatened_nodes[i]
        threatened_nodes[i] = threatened_nodes[j]
        threatened_nodes[j] = v
        
    # # print safe_nodes
    # safe_zone = safe_nodes[random.randint(0, len(safe_nodes))]
    
    escape_routes = []
    for node in threatened_nodes[:num_evacuations]:
        route = treeify( nx.dijkstra_path(G, node, safe_zone, weight='criterion'), escape_routes )
        escape_routes.append( route )
        # print escape_routes[-1]
            
    
    town = {}.fromkeys(threatened_nodes[:num_evacuations])
    
    for node in threatened_nodes[:num_evacuations]:
        town[node] = random.uniform(0,1)
    
    population = {}.fromkeys(range(num_evacuations))
    maximum_rate = {}.fromkeys(range(num_evacuations))
    for evacuee in range(num_evacuations):
        population[evacuee] = random_population()
        maximum_rate[evacuee] = population[evacuee]
        
    
    tightest_deadline = {}.fromkeys(range(num_evacuations))
    relevant_arcs = set([])
    capacities = {}
    arcs = {}
    for evacuee, route in zip(range(num_evacuations), escape_routes):
        tightest_deadline[evacuee] = sys.maxint
        # deadline_arc = None
        l = 0
        for u,v in zip(route[:-1], route[1:]):
            if G.edges[u,v]['duedate'] != sys.maxint:                    
                if tightest_deadline[evacuee] > (time_factor * G.edges[u,v]['duedate'] - l):
                    tightest_deadline[evacuee] = (time_factor * G.edges[u,v]['duedate'] - l)
                    # deadline_arc = (u,v)
            if not arcs.has_key((u,v)):
                arcs[(u,v)] = [(evacuee,l)]
                capacities[(u,v)] = -1
            else:
                arcs[(u,v)].append((evacuee,l))
            l += int(G.edges[u,v]['distance'] / SPEED) 
        # relevant_arcs.add( deadline_arc )
        
                
    for route,evacuee in zip(escape_routes, range(num_evacuations)):
        min_Q = sys.maxint
        N = len(arcs[(route[-2], route[-1])])
        arc_of = [[] for i in range(N)]
          
        for u,v in zip(route[:-1], route[1:]):
            arc_of[len(arcs[(u,v)])-1].append((G.edges[u,v]['capacity'],u,v))
        for n in reversed(range(N)):
            if len(arc_of[n]) > 0:
                Q,u,v = min(arc_of[n])
                if Q < min_Q and capacities[(u,v)] == -1:
                    min_Q = Q
                    if n != 0:
                        relevant_arcs.add((u,v))
                        capacities[(u,v)] = Q
                    maximum_rate[evacuee] = min(maximum_rate[evacuee], Q)

       
    outfile = open('%s.evac'%(filename), 'w')

    outfile.write('%i %i\n'%(num_evacuations, len(relevant_arcs)))
    for evacuee in range(num_evacuations):
        outfile.write('%i %i %i\n'%(population[evacuee], maximum_rate[evacuee], tightest_deadline[evacuee]))
    for u,v in relevant_arcs:
        # outfile.write('arc=(%i,%i) '%(u,v))
        if capacities.has_key((u,v)) and len(arcs[(u,v)]) > 1: # when there is a single task, the maximum rate of the task has already been pruned accordingly
            outfile.write('%i'%(capacities[(u,v)]))
        else:
            outfile.write('-1')
        outfile.write(' %i'%(len(arcs[(u,v)])))
        for evacuee,l in arcs[(u,v)]:
            outfile.write(' %i %i'%(evacuee, l))
        outfile.write('\n')
    outfile.close()
        
    return escape_routes
    
    
def random_population():
    return random.randint(100, 5000)
    


secondary_road_flowrate = 70
# #persons/vehicle * #lanes * #vehicles/meter * speed (m/s)
# 3 * 1 * .02 * 20 = 1.2 persons/second = 72 persons/minute

primary_road_flowrate = 130
# #persons/vehicle * #lanes * #vehicles/meter * speed (m/s)
# 3 * 2 * .015 * 25 = 2.25 persons/second = 135 persons/minute

highway_flowrate = 200
# #persons/vehicle * #lanes * #vehicles/meter * speed (m/s)
# 3 * 3 * .0125 * 30 = 3.375 persons/second = 202.5 persons/minute

SPEED = 25 # speed in distance unit / time unit in the graph (50m / 1')


def build_roads(G, pos):
    nationales = []
    for r in range(1,size,size/grid):
        p1 = find_closest(pos, r, 0)
        p2 = find_closest(pos, r, size-1)
        nationales.append( nx.shortest_path(G, source=p1, target=p2, weight='distance') )
      
        p1 = find_closest(pos, 0, r)
        p2 = find_closest(pos, size-1, r)
        nationales.append( nx.shortest_path(G, source=p1, target=p2, weight='distance') )
   
    first = capital(G, ties=lambda x,y: x if closer_to_corner(pos[x], pos[y], size) else y) 
    second = capital(G, constraint=lambda x: (distance(pos[first], pos[x]) > size/2))
    third = capital(G, constraint=lambda x: (distance(pos[first], pos[x]) > size/2 and distance(pos[second], pos[x]) > size/2))

    highway = {}.fromkeys([first, second, third])
    nearest_corner = {}.fromkeys([first, second, third])
    
    for source in [first, second, third]:
        highway[source] = nx.shortest_path(G, source, weight='distance')

    center = None
    smallest = sys.maxint
    no = None
    so = None
    ne = None
    se = None 
    for node in G.nodes:
        
        x,y = pos[node]
        
        if (x <= 0 or x >= size-1) and (y <= 0 or y >= size-1) :
            # print node, pos[node]

            if x == 0:
                if y == 0:
                    so = node
                else:
                    no = node
            else:
                if y == 0:
                    se = node
                else:
                    ne = node
        
        dist = len(highway[first][node]) + len(highway[second][node]) + len(highway[third][node])
        if dist < smallest:
            smallest = dist
            center = node
         
    for source in [first, second, third]:
        nearest_corner[source] = no
        for corner in [so, ne, se]:
            if distance(pos[source], pos[corner]) < distance(pos[source], pos[nearest_corner[source]]):
                nearest_corner[source] = corner                
                
    for e in G.edges:
        G.edges[e]['capacity'] = (secondary_road_flowrate *  (1 + G.edges[e]['distance']/size))
        G.edges[e]['duedate'] = sys.maxint
        G.edges[e]['color'] = 'black'
        # print e, G.edges[e]['distance'], G.edges[e]['capacity']
        
    for road in nationales:
        for e in zip(road[:-1], road[1:]):
            G.edges[e]['capacity'] = (primary_road_flowrate *  (1 + G.edges[e]['distance']/size))
            G.edges[e]['color'] = 'green'
            
    for source in [first, second, third]:
        for e in zip(highway[source][center][:-1], highway[source][center][1:]):
            G.edges[e]['capacity'] = (highway_flowrate * (1 + G.edges[e]['distance']/size))
            G.edges[e]['color'] = 'blue'
        for e in zip(highway[source][nearest_corner[source]][:-1], highway[source][nearest_corner[source]][1:]):
            G.edges[e]['capacity'] = (highway_flowrate * (1 + G.edges[e]['distance']/size))
            G.edges[e]['color'] = 'blue'


def print_fire(G, pos, fire, visible):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, size-1.0)
    ax.set_ylim(0, size-1.0)

    # nx.draw_networkx_nodes(G, pos, node_size=5, with_labels=True, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=1)

    colored_edges = {}
    colored_edges['blue'] = [e for e in G.edges if G.edges[e]['color'] == 'blue' and G.edges[e]['duedate'] == sys.maxint]
    colored_edges['green'] = [e for e in G.edges if G.edges[e]['color'] == 'green' and G.edges[e]['duedate'] == sys.maxint]
    colored_edges['black'] = [e for e in G.edges if G.edges[e]['color'] == 'black' and G.edges[e]['duedate'] == sys.maxint]

    for color in ['blue', 'green', 'black']:
        nx.draw_networkx_edges(G, pos, edge_color=color, width=2, edgelist=colored_edges[color])

    vfire = (fire & visible)
    for x,y in visible:
        if  (x,y) not in fire:
            ax.add_patch( patches.Rectangle((x, y), step, step, facecolor="black" ) )
    for x,y in vfire:
        ax.add_patch( patches.Rectangle((x, y), step, step, facecolor="red" ) )
        
def print_evac(G, pos, escape_routes, safe_zone):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, size-1.0)
    ax.set_ylim(0, size-1.0)

    # nx.draw_networkx_nodes(G, pos, node_size=5, with_labels=True, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=1)

    colored_edges = {}
    colored_edges['blue'] = [e for e in G.edges if G.edges[e]['color'] == 'blue']
    colored_edges['green'] = [e for e in G.edges if G.edges[e]['color'] == 'green']
    colored_edges['black'] = [e for e in G.edges if G.edges[e]['color'] == 'black']

    for color in ['blue', 'green', 'black']:
        nx.draw_networkx_edges(G, pos, edge_color=color, width=2, edgelist=colored_edges[color])
    
    nx.draw_networkx_edges(G, pos, edge_color='red', width=5, edgelist=[e for route in escape_routes for e in zip(route[1:], route[:-1])])
    nx.draw_networkx_nodes(G, pos, node_size=100, with_labels=True, font_weight='bold', node_color='red', nodelist=[route[0] for route in escape_routes])
    nx.draw_networkx_nodes(G, pos, node_size=100, with_labels=True, font_weight='bold', node_color='blue', nodelist=[safe_zone])


def print_road(G, pos):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, size-1.0)
    ax.set_ylim(0, size-1.0)

    # nx.draw_networkx_nodes(G, pos, node_size=5, with_labels=True, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=1)

    colored_edges = {}
    colored_edges['blue'] = [e for e in G.edges if G.edges[e]['color'] == 'blue']
    colored_edges['green'] = [e for e in G.edges if G.edges[e]['color'] == 'green']
    colored_edges['black'] = [e for e in G.edges if G.edges[e]['color'] == 'black']

    for color in ['blue', 'green', 'black']:
        nx.draw_networkx_edges(G, pos, edge_color=color, width=2, edgelist=colored_edges[color])
    
               
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Wildfire simulator and evacuation plan generator')
    parser.add_argument('file',type=str,help='path to road network files (to write in --road mode, to read in --evacuation mode)')
    parser.add_argument('--road', default=False, action='store_true')
    parser.add_argument('--evacuation', default=False, action='store_true')


    parser.add_argument('--speed',type=int,default=25,help='Speed of evacuation (~ m/s)')
    parser.add_argument('--size',type=int,default=1000,help='Size of the land area (x50m)')
    parser.add_argument('--limit',type=int,default=800,help='(Roughly) the nnumber of intersections in the road network')
    parser.add_argument('--grid',type=int,default=4,help='Density of the primary road grid: <int> default=4')
    parser.add_argument('--datefactor',type=int,default=10,help='Priority given to safe edges in evacuation: <int> default=10')
    parser.add_argument('--flowfactor',type=int,default=5,help='Priority given to faster edges in evacuation: <int> default=5')
    parser.add_argument('--increaseflow',type=float,default=1.0,help='Multiplier applied to flow rates: <float> default=1')
    
    parser.add_argument('--intensity',type=float,default=.25,help='Intensity of the fire: [0,1] default=.25')
    parser.add_argument('--step',type=int,default=30,help='Granularity of the fire areas (x50m): <int> default=30')
    parser.add_argument('--num_iter',type=int,default=65,help='Number of wildfire expention steps: <int> default=65')
    parser.add_argument('--firepace',type=int,default=2,help='Time between wildfire expention (in minutes): <int> default=5')
    
    
    parser.add_argument('--num_evacuations',type=int,default=10,help='Number of evacuation zones: <int> default=10')
        
    parser.add_argument('--seed',type=int,default=12345,help='Random seed')


    parser.add_argument('--printfire', default=False, action='store_true')
    parser.add_argument('--printroad', default=False, action='store_true')
    parser.add_argument('--printevac', default=False, action='store_true')
    

    
    args = parser.parse_args()

    # Road network stuff
    size = args.size #(x 50m = 50km)
    grid = args.grid
    limit = args.limit
    SPEED = args.speed
    
    # wildfire stuff
    step = args.step #(x 50m = 1,5km)
    intensity = args.intensity
    num_iter = args.num_iter
    
    # evacuation stuff
    date_factor = args.datefactor
    flow_factor = args.flowfactor
    num_evacuations = args.num_evacuations
        
    secondary_road_flowrate *= args.increaseflow
    primary_road_flowrate *= args.increaseflow
    highway_flowrate *= args.increaseflow    
        
    random.seed(args.seed)
    
    escape_routes = []


    if args.road:
        
        print 'generate roads...',
        sys.stdout.flush()
        
        G, pos = generate_road_network(limit=limit, size=size)
        
        print ' %i nodes, %i edges'%(len(G.nodes), len(G.edges))
        
        print 'built highways...',
        sys.stdout.flush()
        
        build_roads(G, pos)
        
        print ' done'
        
        
        print 'save graph...',
        sys.stdout.flush()
        
        cPickle.dump(G, open('%s.graph'%args.file, 'w'))
        cPickle.dump(pos, open('%s.pos'%args.file, 'w'))
        
        print ' in %s.graph / %s.pos'%(args.file, args.file)


    if args.evacuation:
        
        print 'load graph...',
        sys.stdout.flush()
        
        G = cPickle.load(open('%s.graph'%args.file, 'r'))
        pos = cPickle.load(open('%s.pos'%args.file, 'r'))
    
        nodes = list(G.nodes)
    
        print ' %i nodes, %i edges'%(len(nodes), len(G.edges))
    
        instance_built = False
        while not instance_built:
        
            threatened = set([])
            safe = set(nodes)   
    
            K = 3
        
            foyer = (random.randint(size/K,(K-1)*size/K), random.randint(size/K,(K-1)*size/K))
            fire = set([foyer])
            wind = (random.uniform(-1,1), random.uniform(-1,1)) 
            burnt = set([p for p in fire])
            visible = set([p for p in fire])
        
            direction = (0,0)

        
            print 'simulate fire',
            sys.stdout.flush()
        
            date = 0
            niters = 1
            for T in range(num_iter):
                niters = niters*1
        
                # print niters
        
                # x,y = wind
                # x += random.uniform(-.2,.2)
                # y += random.uniform(-.2,.2)
                # wind = x,y
        
                for t in range(niters):
            
                    # print 'iter %i'%date,
                    # sys.stdout.flush()
            
                    fire,burnt,visible,direction = burn(fire, burnt, visible, wind=wind, step=step, intensity=intensity, foyer=foyer, direction=direction)
            
                    V = list(G.nodes)
            
                    # print '%i burnt, %i on fire'%(len(burnt), len(fire)),
                    # sys.stdout.flush()
            

                    (xmin,xmax), (ymin,ymax) = box_hull(burnt) 
                     
                    for v in V:
                        xv,yv = pos[v]
                        if xv >= xmin and xv <= xmax+step and yv >= ymin and yv <= ymax+step:
                            for x,y in fire:
                                if xv >= x and xv <= x+step and yv >= y and yv <= y+step:
                                    # G.remove_node(v)
                                    threatened.add(v)
                                    for u in G[v]:
                                        G.edges[u,v]['duedate'] = date
                                    # safe.remove(v)
                                    break;

                    vfire = (fire & visible)
                    for (u,v) in list(G.edges):
                        if G.edges[u,v]['duedate'] == sys.maxint and intersect( xmin, xmax, ymin, ymax, pos[u], pos[v] ):
                            for x,y in vfire:
                                if intersect( x, x+step, y, y+step, pos[u], pos[v] ):
                                    # G.remove_edge(u,v)
                                    G.edges[u,v]['duedate'] = date
                                    break;
                                
                    # print_fire(G, pos, fire, visible)
                
                    date += 1
                    sys.stdout.write('.')
                    sys.stdout.flush()


            safe -= threatened
    
            safe_nodes = list(safe)
            threatened_nodes = list(threatened)
        
            print ' %i burnt nodes, %i safe nodes'%(len(threatened_nodes), len(safe_nodes))
        
            if len(threatened_nodes) >= num_evacuations and len(safe_nodes) > 0:
                max_capacity = max([G.edges[e]['capacity'] for e in G.edges])
                max_distance = max([G.edges[e]['distance'] for e in G.edges])
    
                edges = list(G.edges)
                for e in edges:
                    crit = 0
                    d = G.edges[e]['duedate']
                    if d == sys.maxint :
                        d = num_iter
            
                    crit += float(date_factor*((num_iter-d)))/float(num_iter)  
                    ccrit = float(max_capacity-G.edges[e]['capacity'])/float(max_capacity)
                    dcrit = float(G.edges[e]['distance'])/float(max_distance)                
                    crit += float(flow_factor)*ccrit*dcrit

                    G.edges[e]['criterion'] = crit
            
                # target = (foyer[0]-size*direction[0], foyer[1]-size*direction[1])
                safe_zone = find_closest(pos, foyer[0]-size*direction[0], foyer[1]-size*direction[1], nodes=safe_nodes)
          
                print 'compute escape routes...',
                sys.stdout.flush()
            
                escape_routes = write_evacuation_plan(G, threatened_nodes, safe_zone, num_evacuations, args.firepace, 'data/%s_%i_%i_%i_%i'%(args.file, num_evacuations, SPEED, args.firepace, args.seed))
          
                print ' instance saved in data/%s_%i_%i_%i_%i'%(args.file, num_evacuations, SPEED, args.firepace, args.seed)
          
                instance_built = True;
                # print foyer # x,y
                # print direction # -dx,-dy
            else:
                print 'no instance created!'
        
        

        
        

    if args.printfire:
        print_fire(G, pos, fire, visible)

    if args.printroad:
        print_road(G, pos)
        
    if args.printevac:
        print_evac(G, pos, escape_routes, safe_zone)

    if args.printfire or args.printroad:
        plt.show()


    
    



