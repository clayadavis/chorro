# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
"""
This is the core module for geoCPM. Check out the test_drive notebook 
for an interactive implementation, or of course check out the webapp.
"""

# <codecell>

from numba.decorators import jit, autojit
from numba import float64, i4, i1
import numpy as np
import pandas as pd
import shapefile #This is a modified version. Patches will be submitted upstream.
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import networkx as nx
import cProfile as profile
import copy, itertools, time, random

# <codecell>
def load_shapefile(fname):
    sf = shapefile.Reader(fname, quiet=True)
    return sf.shapeRecordDataFrame()
    # <codecell>

def rescale(pts, x_min=0.0, y_min=0.0, x_ar=1.0, y_ar=1.0):
    "Expects an Nx2 input array of points."
    x_col = np.round((pts[:,0] - x_min)*x_ar)
    y_col = np.round((pts[:,1] - y_min)*y_ar)
    return np.vstack((x_col, y_col)).T

def base_256(n):
    ni = int(n)
    return (ni//65536, ni//256, ni%256)

def rgb_to_index(arr):
    return arr[:,:,0]*65536 + arr[:,:,1]*256 + arr[:,:,2]

def draw_map(df, color=base_256, bg=(255, 255, 255), x_dim=1000):
    x_vals = [df['shape', 'bbox'][x][0:3:2] for x in df.index]
    y_vals = [df['shape', 'bbox'][x][1:4:2] for x in df.index]
    x_max = max(max(x_vals))
    x_min = min(min(x_vals))
    y_max = max(max(y_vals))
    y_min = min(min(y_vals))
    aspect_ratio = (x_max-x_min)/(y_max-y_min)
    x_dim = x_dim
    y_dim = int(round(x_dim/aspect_ratio))
    (x_ar, y_ar) = (x_dim/(x_max-x_min), y_dim/(y_max-y_min))
    im = Image.new('RGB', (x_dim, y_dim), bg) # Create a blank image
    draw = ImageDraw.Draw(im)
    for idx in range(len(df)):
        rescaled_poly = rescale(np.array(df['shape', 'points'][idx]), x_min, y_min, x_ar, y_ar)
        draw.polygon(rescaled_poly.flatten().tolist(), fill=color(idx))
        #draw.polygon(rescaled_poly.flatten().tolist(), fill="magenta", outline="black")
    return im

# <codecell>
def generate_grid(data_frame):
    tic = time.time()
    im = draw_map(data_frame)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    print "Drawing took %0.3f seconds." % (time.time() - tic)
    #im.show()
    #plt.imshow(im)
    grid = rgb_to_index(np.array(im))
    return grid

# <codecell>

@jit(argtypes=[i4[:,:], i4], restype=i4[:,:])
def jit_get_neighbor_weights(grid, N):
    weights = np.zeros((N, N), dtype=np.uint32)
    for r in xrange(grid.shape[0]):
        for c in xrange(grid.shape[1]):
            if grid[r, c] != 16777215:
                my_id = grid[r,c]
                if r-1 >= 0             and grid[r-1,c] != my_id and grid[r-1, c] != 16777215:
                    weights[grid[r-1,c], my_id] += 1
                    
                if r+1 < grid.shape[0]  and grid[r+1,c] != my_id and grid[r+1, c] != 16777215:
                    weights[grid[r+1,c], my_id] += 1
                    
                if c-1 >= 0             and grid[r,c-1] != my_id and grid[r, c-1] != 16777215:
                    weights[grid[r,c-1], my_id] += 1
                    
                if c+1 < grid.shape[1]  and grid[r,c+1] != my_id and grid[r, c+1] != 16777215:
                    weights[grid[r,c+1], my_id] += 1
    return weights

def make_neighbor_graph(weights):
    G = nx.Graph()
    #graph_label = lambda x: "%s_%s" % (df['record', 'COUNTY'][x], df['record', 'MCD'][x])
    #graph_label = lambda x: df['record', 'COUSUBFP'][x]
    graph_label = lambda x: x
    for (a, b) in zip(*weights.nonzero()):  
        w = weights[a,b]+weights[b,a]
        if w > 0:
            G.add_edge(graph_label(a), graph_label(b), weight=w)
    return G

# <codecell>

def make_graph(data_frame):
    grid = generate_grid(data_frame)
    tic = time.time()
    network_weights = jit_get_neighbor_weights(grid, len(data_frame))
    print "Getting neighbor weights took %0.3f seconds." % (time.time() - tic)

    tic = time.time()
    neighbor_graph = make_neighbor_graph(network_weights)
    ## This works, but doesn't look up the labels.
    #G = nx.from_numpy_matrix(nw) 
    print "Making neighbor graph took %0.3f seconds." % (time.time() - tic)
    return neighbor_graph

# <codecell>

class SimulationBase(object):
    def __init__(self, G, df, K, **kwargs):
        self.G = G.copy()
        self.K = K
        self.df = df.copy()
        self.mcs = 0
        self.T = kwargs.get('T', 1)
        self.changes_accepted = []
        self.E_old = None
        
        self.seed_groups()
        self.groups = np.array([self.G.node[n].get('group',0) 
                            for n in sorted(self.G.nodes())], dtype=np.int8)
        
        self.border_cells = np.zeros(len(self.df), dtype=np.bool)
        self._update_border_status()
        
        self.weights = np.array([(u,v,w['weight']) for (u,v,w) in self.G.edges(data=True)], dtype=np.int32)
        #self.adj = np.array(nx.to_numpy_matrix(self.G, weight='weight'), dtype=uint32)
        
    def randomize_groups(self):
        for n in self.G:
            self.G.node[n]['group'] = random.randint(0,self.K-1)
        
    def seed_groups(self):
        while True:
            ungrouped = set(self.G.nodes())
            grouped = set()
            def assign(n, group_id):
                self.G.node[n]['group'] = group_id
                ungrouped.remove(n)
                grouped.add(n)
                
            for i in range(self.K):
                n = random.sample(ungrouped, 1).pop()
                assign(n, i)
                
            iterations = 0
            while ungrouped:
                n = random.sample(grouped, 1).pop()
                my_group_id = self.G.node[n]['group']
                for nbr in self.G[n]:
                    if nbr in ungrouped:
                        assign(nbr, my_group_id)
                iterations += 1
            if len(set([self.G.node[n]['group'] for n in self.G])) == self.K:
                break
            else:
                print "Seeding failed, re-trying."
        print "Seeding took %i iterations." % iterations
        
    def connected_groups(self, groups, to_check, G):
        for g_id in to_check:
            member_nodes = (self.groups == g_id).nonzero()[0]
            if len(member_nodes) == 0 or not nx.is_connected(self.G.subgraph(member_nodes)):
                return False
        return True
        
    def draw(self):
        imgplot = plt.imshow(self.get_image())
        imgplot.set_cmap('spectral')
        
    def get_image(self, bg='white', x_dim=1000):
        im = draw_map(self.df, color=lambda x: (1+self.G.node[x]['group'])*250000, bg=bg, x_dim=x_dim)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        return im
        
    def _update_border_status(self, idx_list=None):
        if idx_list == None:
            idx_list = self.G
        for idx in idx_list:
            my_group = self.G.node[idx]['group']
            if True in [self.G.node[nbr]['group'] != my_group for nbr in self.G[idx]]:
                self.G.node[idx]['border'] = True
                self.border_cells[idx] = True
            else:
                self.G.node[idx]['border'] = False
                self.border_cells[idx] = False
    
    def run(self, N=1, debug=False):
        try:
            self.compute_E()
        except AttributeError:
            print '"%s" object has no "compute_E" method. You need to subclass SimulationBase and define one.' % self.__class__
            
        for i in xrange(N):
            self._step(debug)
    
    def _step(self, debug=False):
        self.E_old = None
        self.mcs += 1
        changes_accepted = 0
        #seq = np.random.random_integers(0, len(self.df.index)-1, len(self.df.index))
        seq = np.random.permutation(len(self.df.index))
        for idx in seq:
            if debug >= 2:
                print "id:", idx
            if not self.G.node[idx]['border']:
                if debug >= 2:
                    print "...not border."
                continue
                
            my_group = self.G.node[idx]['group']
            nbr_groups = set([self.G.node[nbr]['group'] for nbr in self.G[idx]])
            singleton = my_group not in nbr_groups
            if not singleton:
                nbr_groups.remove(my_group)
            nbr_groups = list(nbr_groups)

            ## Get E of the state before making changes, use a stored one if we can
            if self.E_old == None:
                E_old = self.compute_E()
            else:
                E_old = self.E_old
            
            ## Make a change
            my_new_group = random.choice(nbr_groups)
            self.G.node[idx]['group'] = my_new_group
            self.groups[idx] = my_new_group
            
            ## It's dangerous not to update border status here, but right now we 
            ## only need to do so if the change is accepted, as neither of the
            ## objectives use it.
            #self._update_border_status([idx]+self.G[idx].keys())
            
            ## Calculate E_new
            E_new = self.compute_E()
            delta_E = E_new - E_old
            if debug >= 2:
                print "E_new: %s\nE_old: %s\ndelta_E: %s\nbzf: %s" % (E_new, E_old, delta_E, np.e**(-delta_E/self.T))
            ## This now includes the singleton and connectedness heuristics!
            if (delta_E < 0 or (random.random() < np.e**(-delta_E/self.T)) or singleton
                ) and self.connected_groups(self.groups, nbr_groups + [my_group], self.G):
                changes_accepted += 1
                if debug >= 2:
                    print "accepted"
                self._update_border_status([idx]+self.G[idx].keys())
                self.E_old = E_new
            else: ## Change things back
                self.G.node[idx]['group'] = my_group
                self.groups[idx] = my_group
                #self._update_border_status([idx]+self.G[idx].keys())
                if debug >= 2:
                    print "rejected"
                ##self.E_old is still good
        self.changes_accepted.append(changes_accepted)
        
# <headingcell level=3>

# A SimulationBase doesn't have an energy function built in. We need to extend it and add those.

# <codecell>

class Simulation(SimulationBase):
    ## Can redefine and extend base methods.
    def __init__(self, *args, **kwargs):
        self.lambda_comp, self.lambda_pop = kwargs['lambdas']
        self.populations = np.array(self.df['record', 'POP2000'], dtype=np.int32)
        SimulationBase.__init__(self, *args, **kwargs)
    
    def run(self, N=1, debug=False):
        if debug:
            print "%s\t%s\t%s\ttotal" % ('MCS', 'Pop', 'Compact')
        SimulationBase.run(self, N, debug)
    
    def _step(self, debug):
        SimulationBase._step(self, debug)
    ##    or super(Simulation, self)._step(debug)
        if debug:
            pe = self.jit_population_equality(self.groups, self.K, self.populations)
            comp = self.jit_vector_compactness(self.groups, self.weights)
            print "%0.3i\t%ix%i\t%ix%i\t%i" % (self.mcs, self.lambda_pop, pe, 
                                    self.lambda_comp, comp, self.compute_E())
            
    @staticmethod
    @jit(argtypes=[i1[:], i4[:,:]], restype=i4)
    def jit_vector_compactness(groups, weights):
        J = np.zeros(weights.shape[0], dtype=np.uint32)
        for r in xrange(weights.shape[0]):
            if groups[weights[r,0]] != groups[weights[r,1]]:
                J[r] = weights[r,2]
        return J.sum()
    
    @staticmethod
    @jit(argtypes=[i1[:], i1, i4[:]], restype=float64)
    def jit_population_equality(groups, K, pops):
        population_vector = np.zeros(K, dtype=np.uint32)
        for i in xrange(len(groups)):
            population_vector[groups[i]] += pops[i]
        return population_vector.std()
    
    def compute_E(self):
        return self.lambda_comp * self.jit_vector_compactness(
                                        self.groups, self.weights) \
             + self.lambda_pop * self.jit_population_equality(
                                    self.groups, self.K, self.populations)

