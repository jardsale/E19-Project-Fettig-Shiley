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

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

URL = "https://maps.googleapis.com/maps/api/elevation/json?locations="

def main():
    lat = 39.7391536
    long = -104.9847034
    loadData(lat, long)

def getData(p1, p2):
    """
    p1, p2: tuples of x,y coordinates to find the distance between
    """
    return None 


def loadData(lat, long):
    res = requests.get(f"{URL}{lat}%2C{long}&key={GOOGLE_API_KEY}")
    print(res.json())

main()