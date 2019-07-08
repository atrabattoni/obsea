"""
AIS module.

Used to read AIS files.

"""
import pandas as pd
import datetime as dt


def read_cls(fname, cargo_and_tanker=True):
    """
    Read AIS data from CLS format.

    Parameters
    ----------
    fname : string
        File name.
    cargo_and_tanker : bool, optional
        Whether only to keep cargos and thankers.

    Returns
    -------
    pandas.DataFrame
        AIS data as a DataFrame with four columns: mmsi - lon - lat - timestamp
        (POSIX timestamps in seconds)

    """
    ais = pd.read_csv(fname, sep=';')

    if cargo_and_tanker:
        mask = ((ais['aisShipType'] >= 70)
                & (ais['aisShipType'] < 90)
                & (ais['navigationStatus'] == 0))
        ais = ais[mask]

    def parser(x):
        return dt.datetime(
            int(x[6:10]),
            int(x[3:5]),
            int(x[0:2]),
            int(x[11:13]),
            int(x[14:16]),
            int(x[17:19]))

    s = ais.locDate + ' ' + ais.locTime
    ais.loc[:, 'timestamp'] = s.apply(parser)

    ais = ais[['mmsi', 'lon', 'lat', 'timestamp']]

    return ais


def read_marine_traffic(fname, terrestrial=True):
    """
    Read AIS data from Marine Traffic format.

    Parameters
    ----------
    fname : string
        File name.
    terrestrial : bool, optional
        Whether only to keep terrestrial AIS.

    Returns
    -------
    pandas.DataFrame
        AIS data as a DataFrame with four columns: mmsi - lon - lat - timestamp
        (POSIX timestamps in seconds)

    """
    ais = pd.read_csv(fname, parse_dates=['TIMESTAMP UTC'])

    if terrestrial:
        mask = (ais['STATION'] == 'TER') & (ais['STATUS'] == 0)
        ais = ais[mask]

    ais = ais.drop(columns=['STATION', 'STATUS'])
    ais = ais.rename(columns={
        'MMSI': 'mmsi',
        'SPEED': 'sog',
        'COURSE': 'cog',
        'HEADING': 'heading',
        'LAT': 'lat',
        'LON': 'lon',
        'TIMESTAMP UTC': 'timestamp'})

    ais = ais[['mmsi', 'lon', 'lat', 'timestamp']]

    return ais


def select_ships(ais, mmsi_list):
    """
    Filter out ship from ais which do not belongs to mmsi_list.

    Parameters
    ----------
    ais : pandas.DataFrame
        AIS data.
    mmsi_list : list
        List of wanted ships' MMSI.

    Returns
    -------
    pandas.DataFrame
        AIS data containing ship in mmsi_list.

    """
    mask = ais['mmsi'].isin(mmsi_list)
    ais = ais[mask]
    return ais
