"""Station module."""
from obspy.core.inventory import Inventory, Network


def select_stations(inventory, station_list):
    """
    Select station within an Inventory according to a list.

    Parameters
    ----------
    inventory : obspy.Inventory
        The inventory.
    station_list : TYPE
        The station list.

    Returns
    -------
    obspy.Inventory
        An inventory only containing stations in station_list.

    """
    if station_list is None:
        return inventory
    network, = inventory
    stations = [station for station in network if station.code in station_list]
    network = Network(code='YV', stations=stations)
    inventory = Inventory(networks=[network], source='')
    return inventory
