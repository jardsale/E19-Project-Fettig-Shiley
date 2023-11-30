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

def main():
    lat = 39.7391536
    long = -104.9847034
    # coord_grid = generateGrid((39.902763, -75.350705), (39.907570, -75.358029))
    coord_grid = loadGrid("swarthmore_elev_map.txt")
    plotGrid(coord_grid)

def plotGrid(coord_grid):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    x_len = len(coord_grid)
    y_len = len(coord_grid[0])
    all_x = []
    all_y = []
    all_z = []

    for x in range(x_len):
        for y in range(y_len):
            ax.scatter3D(x, y, coord_grid[x][y], c = 'b')
            all_x += [x]
            all_y += [y]
            all_z += [float(coord_grid[x][y])]

    X, Y, Z = curveFit(all_x, all_y, all_z, x_len, y_len)
    ax.plot_surface(X, Y, Z, color='red', alpha=0.5) 
    plt.show()

def curveFit(all_x, all_y, all_z, x_len, y_len):
    def func(xy, a, b, c, d, e, f, g, h, i, j, k, l, m, n): 
        x, y = xy
        return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + \
        i*x**2*y + j*y**2*x + n*np.e**(x*k) + m*np.e**(y*l)

    popt, pcov = curve_fit(func, (all_x, all_y), all_z, maxfev = 100000) 

    x_range = np.linspace(0, x_len - 1, 50) 
    y_range = np.linspace(0, y_len - 1, 50) 
    X, Y = np.meshgrid(x_range, y_range) 

    Z = func((X, Y), *popt)
    return X, Y, Z

def loadGrid(file_name):
    # read file 
    f = open(file_name, "r")
    coord_grid = []

    # load data
    for line in f:
        coord_grid.append([float(i) for i in line.strip("\n").split(" ")])
    f.close()

    return coord_grid

def generateGrid(p1, p2):
    """
    p1, p2: tuples of x,y coordinates to find the distance between
    """
    # the largest number of tiles a side can have, for now 
    # maximum 100 squares if both sides are same length, this may change later
    tiles = 10

    x_diff = abs(p1[0]-p2[0])
    y_diff = abs(p1[1]-p2[1])
    dxy = max(x_diff, y_diff)/tiles
    # will exclude the last value of the smaller difference

    # setting up loop variables
    x_iters = int(np.ceil(x_diff/dxy))
    y_iters = int(np.ceil(y_diff/dxy))
    curr_x = p1[0]
    curr_y = p1[1]
    dir_x = int(np.sign(p2[0]-p1[0]))
    dir_y = int(np.sign(p2[1]-p1[1]))

    #initialize grid of correct size
    coord_grid = np.zeros((y_iters, x_iters))

    # generate grid (oriented top left to bottom right with respect to coordinates)
    for i in range(x_iters):
        for j in range(y_iters):
            coord_grid[j*dir_y][i*dir_x] = elevationPoint(curr_x, curr_y) 
            curr_y += dxy * dir_y
        curr_x += dxy * dir_x

    return coord_grid


def elevationPoint(lat, long):
    res = requests.get(f"{URL}{lat}%2C{long}&key={GOOGLE_API_KEY}")
    return res.json()['results'][0]['elevation']

main()