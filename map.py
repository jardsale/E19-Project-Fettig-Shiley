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
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
URL = "https://maps.googleapis.com/maps/api/elevation/json?locations="

class Map:

    def __init__(self, mode, fname, p1 = None, p2 = None, tiles = 10):
        self.mode = mode
        self.fname = fname
        self.p1 = p1
        self.p2 = p2

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
                print(curr_x, curr_y)
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

        def func(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n): 
            x, y = xy
            return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + \
            i*x**2*y + j*y**2*x + n*np.e**(x*k) + m*np.e**(y*l)

        def curveFit(all_x, all_y, all_z, x_len, y_len):
            popt, pcov = curve_fit(func, (all_x, all_y), all_z, maxfev = 100000000, method="trf") 

            x_range = np.linspace(0, x_len - 1, 50) 
            y_range = np.linspace(0, y_len - 1, 50) 
            X, Y = np.meshgrid(x_range, y_range) 

            Z = func((X, Y), *popt)
            return X, Y, Z

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
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

        X, Y, Z = curveFit(all_x, all_y, all_z, x_len, y_len)
        ax.plot_surface(X, Y, Z, color='red', alpha=0.5) 
        plt.show()