from map import Map

def main():
    # swarthmore_map = Map("load", "swarthmore_elev_map_smaller.txt")
    # swarthmore_map.plotGrid()
    # swarthmore_map_smallest = Map("generate", "swarthmore_elev_map_smallest.txt", \
    #  (39.902763, -75.350705), (39.907570, -75.358029), 5)
    # swarthmore_map_smallest = Map("load", "swarthmore_elev_map_smallest.txt", method = "fit")
    # swarthmore_map_smallest.plotGrid()
    # shenandoah_mountain = Map("generate", "shenandoah.txt", \
    #  (38.370437, -78.487635), (38.382601, -78.498038), 10)
    # shenandoah_mountain.plotGrid()
    # shenandoah_mountain = Map("load", "shenandoah.txt")
    # shenandoah_mountain.plotGrid()

    # mt_frances = Map("generate", "frances.txt", \
    #  (63.002348, -151.194851), (62.974696, -151.148923), 15)
    # st_helens = Map("generate", "st_helens_smaller.txt", \
    # (46.182636, -122.232265), (46.235618, -122.146091), 10)
    # mt_frances = Map("load", "frances.txt")
    # mt_frances.plotGrid()
    # hood = Map("generate", "mt_hood.txt", \
    # (45.350626, -121.734667), (45.393944, -121.665809), 20)
    hood = Map("load", "mt_hood.txt", method="fit", path=True)
    hood.plotGrid()
main()