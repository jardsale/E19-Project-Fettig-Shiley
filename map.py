"""
To call google elevation API and generate a "mountain" or other terain using
either curve fitting or polynomial interpolation.

Will likely create a Map class or something similar later

pip libraries:
pip install python-dotenv
pip install requests
"""

import requests 
import os
import time
from dotenv import load_dotenv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import scipy.interpolate 


load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
URL = "https://maps.googleapis.com/maps/api/elevation/json?locations="

class Map:

    def __init__(self, mode, fname, p1 = None, p2 = None, tiles = 10, method="fit"):
        self.mode = mode
        self.fname = fname
        self.p1 = p1
        self.p2 = p2
        self.method = method

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
        """
        p1, p2: tuples of x,y coordinates to find the distance between
        """

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

        def func2(xy, a, b):
            x, y = xy
            return a*x + b*y
        
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

            # define ranges and generate mesh grid
            x_range = np.linspace(0, x_len - 1, 1000)
            y_range = np.linspace(0, y_len - 1, 1000)
            X, Y = np.meshgrid(x_range, y_range)

            # apply the function with parameters found from curve fit to mesh grid
            Z = func20((X, Y), *popt)

            """ Getting Absolute / Percent Error"""
            err_accum = 0

            for i in range(len(all_x)):
                z_approx = func20((all_x[i], all_y[i]), *popt)
                err_accum += abs((all_z[i] - z_approx)/z_approx)

            perc_err = err_accum*100/len(all_x)
            print("Percent Error: %.3f%%" % perc_err)
            abs_err = err_accum / len(all_x)
            print("Absolute Error: %.2fm" % abs_err)

            return X, Y, Z
        
        def interpolate(all_x, all_y, all_z, x_len, y_len):

            # define lienar spaces to operate over
            x_lin = np.linspace(0, x_len - 1, 1000)
            y_lin = np.linspace(0, y_len - 1, 1000)

            # generate mesh grid and cubic spline over all data points
            X, Y = np.meshgrid(x_lin, y_lin, indexing='xy')
            spline = sp.interpolate.Rbf(all_x,all_y,all_z,function='cubic')

            # plot spline on mesh grid
            Z = spline(X, Y)

            return X, Y, Z

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
            X, Y, Z = interpolate(all_x, all_y, all_z, x_len, y_len)
        time_end = time.time()

        # post-processing and plot
        print("Processing Time: %.6f seconds" % (time_end-time_start))
        if(self.method == "fit"):
            ax.plot_surface(X, Y, Z, color='red', alpha=0.5) 
        else:
            # ax.plot_wireframe(X, Y, Z)
            ax.plot_surface(X, Y, Z, alpha=0.8)
        
        plt.show()