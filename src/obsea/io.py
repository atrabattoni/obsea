"""
IO module.

Used to load stream corresponding to a track.

"""
from obspy import UTCDateTime


def load_stream(track, client, inventory, station, channel, min_duration=600,
                max_duration=86400, nb_channels=4, ttol=1e-4):
    """
    Load a stream over the same duration than a track.

    Parameters
    ----------
    track : LineString
        Moving source trajectory.
    client : obpsy.Client
        Used to (down)load the stream.
    inventory : obspy.Inventory
        Used to attach the instrument responses.
    station : obspy.station
        Wanted stations.
    channel : string
        Wanted channels.
    min_duration : int, optional
        Return None if stream duration is too small.
    max_duration : int, optional
        Return None if stream duration is too big.
    nb_channels : int, optional
        Return None if stream number of channel do not match what is expected.
    ttol : float, optional
        Verify that all starttime and endtime of all trace are close enought to
        what was specified.

    Returns
    -------
    obspy.Stream or None
        Loaded seismological data. None if importation failed.

    """
    starttime = UTCDateTime(track.coords[0][-1])
    endtime = UTCDateTime(track.coords[-1][-1])
    duration = endtime - starttime
    if duration > max_duration:
        return None
    st = client.get_waveforms(
        'YV', station.code, '*', channel,
        starttime=starttime,
        endtime=endtime)
    st.merge(1)
    st.trim(starttime, endtime)
    if not st.count() == nb_channels:
        return None
    starttime = min([tr.stats.starttime for tr in st])
    endtime = max([tr.stats.endtime for tr in st])
    duration = endtime - starttime
    if duration < min_duration:
        return None
    st.trim(starttime, endtime)
    if max(abs(tr.stats.starttime - starttime) for tr in st) > ttol:
        return None
    if max(abs(tr.stats.endtime - endtime) for tr in st) > ttol:
        return None
    for tr in st:
        tr.stats.starttime = starttime
    for tr in st:
        if (tr.data == tr[0]).all():
            return None
    st.attach_response(inventory)
    return st
