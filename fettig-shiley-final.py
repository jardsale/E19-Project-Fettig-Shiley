"""
Nicholas Fettig and Matthew Shiley
Engr 019 Fall 2023 Final Project

Our program collects and loads topographical data using the google maps API.
We then approximate what the terrain around these points looks like using 
both a curve fit method and a cubic spline interpolation method. Finally,
we optimize a trail over the approximating function based on an objective
function with several constraints.

Scroll to the bottom (main function) to see tests of our Map. 

Note: I'm attaching a google maps API key too. I have unlimited queries for 
the elevation API so don't worry about any limits! 
"""

import requests 
import time
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.optimize import minimize

GOOGLE_API_KEY = "AIzaSyDCGxhqJ6kPjxhClSMw9bStaNVu-r1uAvg"
URL = "https://maps.googleapis.com/maps/api/elevation/json?locations="

class Map:
    def __init__(self, mode, fname, p1 = None, p2 = None, start = (0,0), end = (10,6), tiles = 10, method="fit", path=False):
        self.mode = mode
        self.fname = fname
        self.p1 = p1
        self.p2 = p2
        self.start = start
        self.end = end
        self.method = method
        self.f = None
        self.path = path
        self.coeffs = []

        if(mode == "load"):
            self.coord_grid, self.tiles = self.loadGrid()
        
        if(mode == "generate"):
            self.tiles = tiles
            self.coord_grid = self.generateGrid()
    
    def loadGrid(self):
        # read file 
        f = open(self.fname, "r")
        coord_grid = []
        f_len = 0

        # load data
        for line in f:
            f_len+=1
            coord_grid.append([float(i) for i in line.strip("\n").split(" ")])
        f.close()
        return coord_grid, max(len(line), f_len)

    def generateGrid(self):
        def elevationPoint(lat, long):
            res = requests.get(f"{URL}{lat}%2C{long}&key={GOOGLE_API_KEY}")
            return res.json()['results'][0]['elevation']

        x_diff = abs(self.p1[0]-self.p2[0])
        y_diff = abs(self.p1[1]-self.p2[1])
        dxy = max(x_diff, y_diff)/self.tiles

        # setting up loop variables
        x_iters = int(np.ceil(x_diff/dxy))
        y_iters = int(np.ceil(y_diff/dxy))
        curr_x = self.p1[0]
        curr_y = self.p1[1]
        dir_x = int(np.sign(self.p2[0]-self.p1[0]))
        dir_y = int(np.sign(self.p2[1]-self.p1[1]))

        #initialize grid of correct size
        coord_grid = np.zeros((y_iters, x_iters))

        # generate grid
        for i in range(x_iters):
            x_idx = i
            curr_y = self.p1[1]
            if(dir_x == -1):
                x_idx = x_iters - i - 1
            for j in range(y_iters):
                # print(curr_x, curr_y)
                y_idx = j
                if(dir_y == -1):
                    y_idx = y_iters - j - 1
                coord_grid[y_idx][x_idx] = elevationPoint(curr_x, curr_y) 
                curr_y += (dxy * dir_y)
            curr_x += (dxy * dir_x)

        # write to a new file
        file_str = ""
        for line in coord_grid:
            file_str += " ".join(str(i) for i in line) + "\n"

        if(os.path.isfile(self.fname)):
            f = open(self.fname, "w")
        else:
            f = open(self.fname, "x")

        f.write(file_str.rstrip("\n"))
        f.close()

        return coord_grid

    def plotGrid(self):

        """define functions of different size parameters"""
        def func4(xy, a, b, c, d):
            x, y = xy
            return a*x*y + b*x + c*y + d
        
        def func10(xy, a, b, c, d, e, f, g, h, i, j, ): 
            x, y = xy
            return a + b*x + c*y + d*x*y + e*x**2 + f*y**2 + g*x*y**2 + h*x**2*y + \
            i*x**3 + j*y**3

        def func20(xy, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20):
            x, y = xy
            return a1 + a2*x + a3*y + a4*x*y + a5*x**2 + a6*y**2 + a7*x*y**2 + a8*x**2*y + \
            + a9*x**2*y**2 + a10*x**3 + a11*y**3 + a12*x*y**3 + a13*x**3*y + a14*x**2*y**3 + a15*x**3*y**2 + \
            a15*x**3*y**3 + a16*x**4 + a17*y**4 + a18*x*y**4 + a19*x**4*y + a20*x**2*y**4

        def curveFit(all_x, all_y, all_z, x_len, y_len):
            # curve fit using function and all datapoints
            popt, pcov = curve_fit(func20, (all_x, all_y), all_z, maxfev = 100000000)
            self.coeffs = popt

            # define ranges and generate mesh grid
            x_range = np.linspace(0, x_len - 1, 1000)
            y_range = np.linspace(0, y_len - 1, 1000)
            X, Y = np.meshgrid(x_range, y_range)

            # apply the function with parameters found from curve fit to mesh grid
            Z = func20((X, Y), *popt)

            """ Getting Absolute / Percent Error"""
            perc_err_accum = abs_err_accum = 0

            for i in range(len(all_x)):
                z_approx = func20((all_x[i], all_y[i]), *popt)
                if(z_approx == 0.0): 
                    continue
                perc_err_accum += abs((all_z[i] - z_approx)/z_approx)
                abs_err_accum += abs(all_z[i] - z_approx)

            perc_err = perc_err_accum*100 / len(all_x)
            print("Percent Error: %.3f%%" % perc_err)
            abs_err = abs_err_accum / len(all_x)
            print("Absolute Error: %.2fm" % abs_err)

            return X, Y, Z
        
        def interp(all_x, all_y, all_z, x_len, y_len):

            # define lienar spaces to operate over
            x_lin = np.linspace(0, x_len - 1, 1000)
            y_lin = np.linspace(0, y_len - 1, 1000)

            # generate mesh grid and cubic spline over all data points
            X, Y = np.meshgrid(x_lin, y_lin, indexing='xy')
            spline = sp.interpolate.Rbf(all_x,all_y,all_z,function='cubic')
            self.f = spline

            # plot spline on mesh grid
            Z = spline(X, Y)

            return X, Y, Z
        
        def function(x, y):
            if len(self.coeffs)>0:
                return func20((x, y), *self.coeffs)

            else:
                return self.f(x, y)

        # helper function to get distance between two points
        def dist(x1, y1, x2, y2):
            return np.sqrt((x1-x2)**2+(y1-y2)**2)

        # path steepness constraint function
        def steepness_constraint(x1, y1, x2, y2, x3, y3, x4, y4):
            xstart = self.start[0]
            ystart = self.start[1]
            xend = self.end[0]
            yend = self.end[1]
            slopes = []
            # Was getting some divide by zero errors, hence the maximum in the denominator. 
            slopes.append((function(x1, y1)-function(xstart, ystart))/max(dist(xstart,ystart,x1,y1), 0.001))
            slopes.append((function(x2, y2)-function(x1, y1))/(max(dist(x1,y1,x2,y2), 0.0001)))
            slopes.append((function(x3, y3)-function(x2, y2))/(max(dist(x2,y2,x3,y3), 0.001)))
            slopes.append((function(x4, y4)-function(x3, y3))/(max(dist(x3,y3,x4,y4), 0.001)))
            slopes.append((function(xend, yend)-function(x4, y4))/(max(dist(x4,y4,xend,yend), 0.001)))
            
            # return the maximum slope between points on the path
            return max(slopes)
        
        # path length constraint function
        def length_constraint(x1, y1, x2, y2, x3, y3, x4, y4):
            xstart = self.start[0]
            ystart = self.start[1]
            xend = self.end[0]
            yend = self.end[1]

            # sum up distances between points on path
            cost = dist(xstart, ystart, x1, y1)
            cost += dist(x1, y1, x2, y2)
            cost += dist(x2, y2, x3, y3)
            cost += dist(x3, y3, x4, y4)
            cost += dist(x4, y4, xend, yend)

            return cost

        # combined cost function with length and steepness constraints
        def path_cost(pts):
            x1, y1, x2, y2, x3, y3, x4, y4 = pts
            lambda1 = 20
            lambda2 = 20

            cost = lambda1 * steepness_constraint(x1, y1, x2, y2, x3, y3, x4, y4)**2
            cost += lambda2 * length_constraint(x1, y1, x2, y2, x3, y3, x4, y4)**2
            return cost

        # define plot and axes
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # preprocessing 
        x_len = len(self.coord_grid)
        y_len = len(self.coord_grid[0])
        all_x = []
        all_y = []
        all_z = []

        for x in range(x_len):
            for y in range(y_len):
                ax.scatter3D(x, y, self.coord_grid[x][y], c = 'b')
                all_x += [x]
                all_y += [y]
                all_z += [float(self.coord_grid[x][y])]

        # fit or interpolate depending on chosen method 
        time_start = time.time()
        
        if self.method == "fit":
            X, Y, Z = curveFit(all_x, all_y, all_z, x_len, y_len)
        else:
            X, Y, Z = interp(all_x, all_y, all_z, x_len, y_len)

        time_end = time.time()

        # if the user has opted to create a path...
        if self.path==True:
            init_guess = []
            for i in range(1,5):
                init_guess += [self.start[0]+(self.end[0]-self.start[0])/5*i, self.start[1]+(self.end[1]-self.start[1])/5*i]
        
            # optimizing the path coordinates (we currently have a fixed number of 6 points)
            path_pts = minimize(path_cost, x0=[1, 1, 2, 2, 3, 3, 3.5, 3.5], bounds=[(0, x_len-1), (0, y_len-1)]*4).x
            
            x = [self.start[0], path_pts[0], path_pts[2], path_pts[4], path_pts[6], self.end[0]]
            y = [self.start[1], path_pts[1], path_pts[3], path_pts[5], path_pts[7], self.end[1]]
            
            # plotting path
            ax.plot(x,y,[function(x[i], y[i]) for i in range(6)])

        # post-processing and plot
        print("Processing Time: %.6f seconds" % (time_end-time_start))
        if(self.method == "fit"):
            ax.plot_surface(X, Y, Z, color='red', alpha=0.5) 
        else:
            # ax.plot_wireframe(X, Y, Z)
            ax.plot_surface(X, Y, Z, alpha=0.8)

        plt.show()

def main():
    """
    Tests using the Map class. This class will create, load, and generate
    plots.

    Guide for plotting between two coordinate points, p1 and p2:

    p1, p2 = (lat, lng)

    Generating --
    Map("generate", fname(str), p1 (tuple), p2(tuple), \
    tile_size(int), method = "fit" / "interp", path = bool)

    Loading -- 
    Map("load", fname(str), method = "fit", path = bool)
    """

    """Swarthmore College"""
    #swarthmore_map = Map("generate", "swarthmore.txt", \
    #(39.902763, -75.350705), (39.907570, -75.358029), 10, method = "fit")
    # swarthmore_map = Map("load", "swarthmore.txt", method = "fit")
    # swarthmore_map.plotGrid()

    """Mount Frances"""
    # mt_frances = Map("generate", "frances.txt", \
    #  (63.002348, -151.194851), (62.974696, -151.148923), 15)
    # mt_frances = Map("load", "frances.txt")
    # mt_frances.plotGrid()

    """Mount Hood"""
    hood = Map("generate", "mt_hood.txt", \
    (45.350626, -121.734667), (45.393944, -121.665809), tiles=15)
    # hood = Map("load", "mt_hood.txt", method="fit", path=True)
    hood.plotGrid()

main()