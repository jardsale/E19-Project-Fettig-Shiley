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
from scipy.optimize import minimize
from matplotlib import cm


load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
URL = "https://maps.googleapis.com/maps/api/elevation/json?locations="

class Map:

    def __init__(self, mode, fname, p1 = None, p2 = None, tiles = 10, method="fit", path=False, start = None, end = None, n_pts = 5):
        self.mode = mode
        self.fname = fname
        self.p1 = p1
        self.p2 = p2

        self.start = start
        self.end = end
        self.end = end
        self.method = method
        self.f = None
        self.path = path
        self.coeffs = []
        self.n_pts = n_pts


        if(mode == "load"):
            self.coord_grid, self.tiles = self.loadGrid()
        
        if(mode == "generate"):
            self.tiles = tiles
            self.coord_grid = self.generateGrid()
        
        if start==None:
            self.start = (0,0)
            self.end = (self.x_len/2, self.y_len/2)
    
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
        self.x_len = len(coord_grid)
        self.y_len = len(coord_grid[0])
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

        self.x_len = x_iters
        self.y_len = y_iters

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

        #Applies a curve fit to topographic data
        def curveFit(all_x, all_y, all_z, x_len, y_len):
            # curve fit using function and all datapoints
            popt, pcov = curve_fit(func20, (all_x, all_y), all_z, maxfev = 1000000000)
            self.coeffs = popt

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
        
        # Interpolates topographic data   
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
        
        # Helper function to calculate distance between two points (x1, y1) and (x2, y2)
        def dist(x1, y1, x2, y2):
            return np.sqrt((x1-x2)**2+(y1-y2)**2)

        def steepness_constraint(x, y):
            xstart = self.start[0]
            ystart = self.start[1]
            xend = self.end[0]
            yend = self.end[1]
            
            cost = abs((function(x[0], y[0])-function(xstart, ystart))/dist(xstart,ystart,x[0],y[0]))
            for i in range(0, self.n_pts-1):
                cost += abs((function(x[i], y[i])-function(x[i+1], y[i+1]))/dist(x[i],y[i],x[i+1],y[i+1]))
            
            cost += abs((function(xend, yend)-function(x[-1], y[-1]))/dist(x[-1],y[-1],xend,yend))
            return cost
        
        def length_constraint(x, y):
            xstart = self.start[0]
            ystart = self.start[1]
            xend = self.end[0]
            yend = self.end[1]

            cost = dist(xstart, ystart, x[0], y[0])
            for i in range(0,self.n_pts-1):
                cost += dist(x[i], y[i], x[i+1], y[i+1])
            
            cost += dist(x[-1], y[-1], xend, yend)
            return cost

        def path_cost(pts):
            x = pts[0:self.n_pts]
            y = pts[self.n_pts:]
            print(x,y)
            lambda1 = 0.1
            lambda2 = 0.2

            cost = lambda1 * steepness_constraint(x, y)**2
            cost += lambda2 * length_constraint(x, y)**2
            return cost

        def initial_guess():
            x0 = []
            for i in range(1,self.n_pts+1):
                x0 += [self.start[0]+(self.end[0]-self.start[0])/(self.n_pts+1)*i]
            for i in range(1,self.n_pts+1):
                x0 += [self.start[1]+(self.end[1]-self.start[1])/(self.n_pts+1)*i]
            return np.array(x0)


        # define plot and axes
        ax = plt.axes(projection='3d')
        
        def function(x, y):
            if len(self.coeffs)>0:
                return func20((x, y), *self.coeffs)
        
            else:
                return self.f(x, y)
            

        # preprocessing 
        all_x = []
        all_y = []
        all_z = []

        for x in range(self.x_len):
            for y in range(self.y_len):
                ax.scatter3D(x, y, self.coord_grid[x][y], c = 'b')
                all_x += [x]
                all_y += [y]
                all_z += [float(self.coord_grid[x][y])]

        # fit or interpolate depending on chosen method 
        time_start = time.time()
        if self.method == "fit":
            X, Y, Z = curveFit(all_x, all_y, all_z, self.x_len, self.y_len)


        else:
            X, Y, Z = interp(all_x, all_y, all_z, self.x_len, self.y_len)

        time_end = time.time()

        # If the user has chosen to generate a path, generate and optimize it
        bs = [(0, self.x_len-1)]*self.n_pts + [(0, self.y_len-1)]*self.n_pts

        if self.path==True:
            path_pts = minimize(path_cost, x0=initial_guess(), 
                                bounds=bs).x
                
        # post-processing and plot
        print("Processing Time: %.6f seconds" % (time_end-time_start))
        if(self.method == "fit"):
            ax.plot_surface(X, Y, Z, color='red', alpha=0.5, cmap=cm.coolwarm) 
        else:
            ax.plot_surface(X, Y, Z, alpha=0.8, cmap=cm.coolwarm)
        
        x = [self.start[0]] + path_pts[:self.n_pts] + [self.end[0]]
        y = [self.start[1]] + path_pts[self.n_pts:] + [self.end[1]]
        ax.plot(x,y,[function(x[i], y[i]) for i in range(self.n_pts)])
        
        plt.show()