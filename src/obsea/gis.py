"""
GIS module.

Used to process AIS data, build ship trajectories and retreive route passing
close to a recording instrument.

"""
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from shapely.geometry import Point, LineString
import xarray as xr


def to_posix(t):
    """
    Convert datetime into posix float in seconds.
    """
    return (t - np.datetime64(1, "s")) / np.timedelta64(1, "s")


def from_ais(ais):
    """
    Convert an AIS dataframe into a track dataarray. 
    """
    keys = ['mmsi', 'imo', 'shipName', 'aisShipType',
            'shipDraught', 'shipLength', 'shipWidth']
    attrs = {key: ais[key].iloc[0] for key in keys if key in ais}
    attrs["crs"] = ccrs.PlateCarree()
    return xr.DataArray(
        data=ais['lon'] + 1j * ais['lat'],
        coords={'time': ais['time']},
        dims='time',
        attrs=attrs,
    )


def to_crs(track, crs):
    """
    Change the CRS of a track.
    """
    x, y, _ = crs.transform_points(
        track.attrs["crs"], track.real.values, track.imag.values).T
    return xr.DataArray(
        data=x + 1j * y,
        coords={'time': track['time']},
        dims='time',
        attrs={'crs': crs},
    )


def to_linestring(track):
    """
    Convert a track into a linestring.
    """
    coords = zip(track.real, track.imag, track['time'])
    return LineString(coords)


def from_linestring(linestring, crs):
    """
    Convert a linestring into a crs.
    """
    x, y, t = [np.asarray(e) for e in zip(*linestring.coords)]
    t = t.astype('datetime64[ns]')
    return xr.DataArray(
        data=x + 1j * y,
        coords={'time': t},
        dims='time',
        attrs={'crs': crs},
    )


def get_distance(linestring, point):
    """
    Compute the distance of a point to a linestring. 
    """
    return point.distance(linestring)


def intersect(linestring, point, radius):
    """
    intersect a linstring with a disk.
    """
    area = point.buffer(radius)
    return area.intersection(linestring)


def sort_linestring(linestring):
    """
    Sort a linestring temporally.
    """
    return LineString(sorted(linestring.coords, key=lambda point: point[-1]))


def read_ais(ais, timedelta):
    """
    Create a Series of tracks from AIS data.

    Parameters
    ----------
    ais : DataFrame
        The AIS data with time stored in the 'time' column
        and position in a 'lon' and 'lat' column.
    timedelta : Timedelta
        The maximum time gap accepted between two AIS logs.

    Returns
    -------
    tracks : Series
        A series containing tracks indexed by their MMSI.

    """
    ais = ais.sort_values(['mmsi', 'time'])
    ais['timedelta'] = ais.groupby('mmsi')['time'].apply(
        lambda x: x.diff())
    ais['passage'] = ais.groupby('mmsi')['timedelta'].apply(
        lambda x: (x > timedelta).cumsum())
    ais = ais.groupby(['mmsi', 'passage']).filter(
        lambda x: len(x) > 1)
    tracks = ais.groupby(['mmsi', 'passage']).apply(
        from_ais)
    tracks = tracks.reset_index('passage', drop=True)
    return tracks


def select_tracks(tracks, station, radius, cpa):
    """
    Select tracks passing close to some interest point.

    Parameters
    ----------
    tracks : GeoSerie
        Tracks from which the selection must be done.
    station : Obspy Station
        Station of interest.
    radius : float
        Tracks are intersected in with a disk with radius is given but that
        value (in metres).
    cpa : float
        Tracks passing farther than this are rejected (in metres).

    Returns
    -------
    df : GeoDataFrame
        The selected tracks in the local azimuthal equidistant projection
        as a GeoDataFrame containing tracks and related statistics.

    """
    crs = ccrs.AzimuthalEquidistant(station.longitude, station.latitude)
    point = Point(0, 0)
    tracks = tracks.apply(to_crs, args=(crs,))
    tracks = tracks.apply(to_linestring)
    tracks = tracks[tracks.apply(get_distance, args=(point,)) <= cpa]
    tracks = tracks.apply(intersect, args=(point, radius))
    tracks = (tracks
              .apply(pd.Series)  # split MultiLineString into LineString
              .stack().reset_index(level=1, drop=True)  # one row per LineString
              .apply(sort_linestring))  # ensure that time is not reversed
    tracks = tracks[tracks.apply(get_distance, args=(point,)) <= cpa]
    tracks = tracks.apply(from_linestring, args=(crs,))
    return tracks


def correct_track(track, B):
    """
    Correct the accoustic center.

    Change the track positions by B in the opposite direction of the ship
    heading. For now works only for rectilinear trajectories.

    Parameters
    ----------
    track : xr.DataArray
        The track to correct.
    B : float
        The amount of correction.

    Returns
    -------
    xr.DataArray
        The corrected track
    """
    v, _ = np.polyfit(to_posix(track["time"]), track.values, 1)
    u = v / np.abs(v)
    track = track - u * B
    return track
