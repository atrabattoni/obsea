"""
GIS module.

Used to process AIS data, build ship trajectories and retreive route passing
close to a recording instrument.

"""
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, LineString
import xarray as xr


def to_posix(t):
    """
    Convert datetime into posix float in seconds.
    """
    return (t - np.datetime64(1, "s")) / np.timedelta64(1, "s")


def from_posix(t):
    """
    Convert posix float into datetime.
    """
    return np.datetime64(int(round(1e9 * t)), "ns")


def from_ais(ais):
    """
    Convert an AIS dataframe into a track dataarray.
    """
    keys = [
        "mmsi",
        "imo",
        "shipName",
        "aisShipType",
        "shipDraught",
        "shipLength",
        "shipWidth",
    ]
    attrs = {key: ais[key].iloc[0] for key in keys if key in ais}
    attrs["crs"] = "proj=lonlat"
    return xr.DataArray(
        data=ais["lon"] + 1j * ais["lat"],
        coords={"time": ais["time"]},
        dims="time",
        attrs=attrs,
    )


def to_crs(track, crs):
    """
    Change the CRS of a track.
    """
    transformer = pyproj.Transformer.from_crs(track.attrs["crs"], crs, always_xy=True)
    x, y = transformer.transform(track.real.values, track.imag.values)
    data = x + 1j * y
    track = track.copy(data=data)
    track.attrs["crs"] = crs
    return track


def to_linestring(track):
    """
    Convert a track into a linestring.
    """
    coords = np.stack((track.real, track.imag, track["time"].astype("float")), axis=1)
    return LineString(coords)


def from_linestring(linestring, crs):
    """
    Convert a linestring into a crs.
    """
    x, y, t = [np.asarray(e) for e in zip(*linestring.coords)]
    t = t.astype("datetime64[ns]")
    return xr.DataArray(
        data=x + 1j * y,
        coords={"time": t},
        dims="time",
        attrs={"crs": crs},
    )


def get_cpa(track):
    linestring = to_linestring(track)
    origin = Point(0, 0)
    cpa = linestring.interpolate(linestring.project(origin))
    x = cpa.coords[0][0]
    y = cpa.coords[0][1]
    t = np.datetime64(int(round(cpa.coords[0][2])), "ns")
    return xr.DataArray([x + 1j * y], {"time": [t]}, "time", attrs=track.attrs)


def simplify(track):
    v, r0 = np.polyfit(to_posix(track["time"]), track.values, 1)
    t = -(r0.real * v.real + r0.imag * v.imag) / np.abs(v) ** 2
    r = v * t + r0
    t = from_posix(t)
    return xr.Dataset(
        data_vars={"r": ("time", [r]), "v": ("time", [v])},
        coords={"time": [t]},
        attrs=track.attrs,
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


def split_multilinestring(geom):
    """ "
    Split any eventual multilinestring into linestrings
    """
    if hasattr(geom, "geoms"):
        geoms = geom.geoms
    else:
        geoms = [geom]
    return pd.Series(geoms)  # TODO: Handle ShapelyDeprecationWarning


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
    ais = ais.sort_values(["mmsi", "time"])
    ais["timedelta"] = ais.groupby("mmsi", group_keys=False)["time"].apply(
        lambda x: x.diff()
    )
    ais["passage"] = ais.groupby("mmsi", group_keys=False)["timedelta"].apply(
        lambda x: (x > timedelta).cumsum()
    )
    ais = ais.groupby(["mmsi", "passage"], group_keys=False).filter(
        lambda x: len(x) > 1
    )
    tracks = ais.groupby(["mmsi", "passage"], group_keys=False).apply(from_ais)
    tracks = tracks.reset_index("passage", drop=True)
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
    crs = f"+proj=aeqd +lon_0={station.longitude} +lat_0={station.latitude}"
    point = Point(0, 0)
    tracks = tracks.apply(to_crs, args=(crs,))
    meta = tracks.apply(getattr, args=("attrs",))
    tracks = tracks.apply(to_linestring)
    tracks = tracks[tracks.apply(get_distance, args=(point,)) <= cpa]
    tracks = tracks.apply(intersect, args=(point, radius))
    tracks = (
        tracks.apply(split_multilinestring)
        .stack()
        .reset_index(level=1, drop=True)  # one row per LineString
        .apply(sort_linestring)
    )  # ensure that time is not reversed
    tracks = tracks[tracks.apply(get_distance, args=(point,)) <= cpa]
    tracks = tracks.apply(from_linestring, args=(crs,))
    meta = meta[tracks.index]
    for attrs, track in zip(meta, tracks):
        track.attrs = attrs
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


def save_complex(data_array, *args, **kwargs):
    ds = xr.Dataset({"real": data_array.real, "imag": data_array.imag})
    return ds.to_netcdf(*args, **kwargs)


def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    return ds["real"] + ds["imag"] * 1j
