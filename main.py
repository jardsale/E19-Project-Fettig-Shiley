from map import Map

def main():
    # swarthmore_map = Map("load", "swarthmore_elev_map_smaller.txt")
    # swarthmore_map.plotGrid()
    # swarthmore_map_smallest = Map("generate", "swarthmore_elev_map_smallest.txt", \
    #  (39.902763, -75.350705), (39.907570, -75.358029), 5)
    # swarthmore_map_smallest.plotGrid()
    # shenandoah_mountain = Map("generate", "shenandoah.txt", \
    #  (38.370437, -78.487635), (38.382601, -78.498038), 10)
    # shenandoah_mountain.plotGrid()
    # shenandoah_mountain = Map("load", "shenandoah.txt")
    # shenandoah_mountain.plotGrid()

    mt_frances = Map("generate", "frances.txt", \
     (63.002348, -151.194851), (62.974696, -151.148923), 15)
    # mt_frances = Map("load", "swarthmore_elev_map_smaller.txt")
    # mt_frances.plotGrid()
    
main()