========
Workflow
========

Here is presented the general workflow used to process seismological data along with AIS archive in Obsea. A Diagram summarize all the interactions. Functions used are elicited along with the module where they are implemented yet all function can be imported directly from the root ``obsea`` module.

.. graphviz::

   digraph {
       node [shape=record];
      "Client" -> "Stream";
      "Stream" -> "Time-Frequency";
      "Time-Frequency" -> "Azigram";
      "Azigram" -> "Time-Azimuth";
      "Time-Frequency" -> "Cepstrogram";
      "AIS" -> "Global Tracks";
      "Global Tracks" -> "Local Tracks";
      "Local Tracks" -> "Orientation-Frequency";
      "Azigram" -> "Orientation-Frequency";
      "Cepstrogram" -> "Post-Processing";
      "Post-Processing" -> "Beamforming";
      "Local Tracks" -> "Beamforming";
      "Inventory" -> "Stream";
      "Inventory" -> "Local Tracks";
      "Local Tracks" -> "Stream" ;
      { rank=same; "Client", "Inventory", "AIS" };
   }

Users are generally asked to provide three type of Inputs:

- **Client**: An ObsPy_ client which will be used to get the seismological data through its ``get_waveforms`` method. Collecting data from data server can be perform through FDSN_ based client. Using local data stored as an SDS_ file-system is also possible.

- **Inventory**: An ObsPy_ Inventory_ which contains information about the stations like its location and its instrumental responses.

- **AIS**: A Pandas DataFrame which must at least have four columns: ``'mmsi'``, ``'lon'``, ``'lat'`` and ``'timestamp'`` (As POSIX timestamps in seconds). Obsea provides helper function to import CSV files of AIS logs from several providers in the ``ais`` module.

Seismological data are handled with the Obspy_ **Stream** class which embed each components as a Trace (documentation can be found here_). Responses must be attached in order to remove the instrumental response. This is can be done by attaching the responses stored in the Inventory to the Stream. If working with source of known trajectories, local tracks ca be provided to the ``load_stream`` (in the ``io`` module) function to load the relevant time segment when the source passes close by the instrument. 

When working with known source positions, **AIS** data is processed into tracks per ship thank to the ``gis`` module. This is done in two steps. First, **Global Tracks** are performed with the ``read_ais`` function. Second, if working with stations, **Local Tracks** in a local coordinate reference system in meters where zero is the instrument location are performed for global tracks passing close enough to the station of interest. This is performed with the ``select_tracks`` function which needs the Inventory to know the instrument location.

In Obsea, all 2D representations are handled thanks to the Xarray_ library. Coordinates must have those names: ``'time'``, ``'frequency'``, ``'azimuth'``, ``'quefrency'``, ``'orientation'``, ``'x'``, ``'y'``. Lets list available 2D representations:

- **Time-Frequecy**: STFT is computed for each Trace in a Stream with the ``time_frequency`` function (in the ``core`` module). If responses are available, instrumental response removal is possible as that stage (water level deconvolution).

- **Azigram**: Horizontal Direction of Arrival is computed with the ``azigram`` function (in the ``core`` module) for each time and frequency from the 4 time-frequency representation of the 4 components of the instruments (The vertical component can be omitted though).

- **Cepstrogram**: Can be performed for any time-frequency representation with the ``cepstrogram`` function of the ``core`` module (usually the pressure channel is used).

- **Time-Azimuth**: Can be performed for any azigram with the ``time-azimuth`` function of the ``core`` module.

- **Orientation-Frequency**: This representation is useful for reorienting OBSs. Need an azigram and a local track. Computed with the ``orientation-frequency`` function of the ``core`` module.

- **Post-Processing**: This is not a 2D representation but many post-processing steps are gathered in the ``processing`` module.

- **Beamforming**: Beamforming in the quefrency domain can be used to retrieve an OBS localization. Local tracks are used as antennas to process cepstrograms with functions in the ``beamforming`` module. 

To see how to implement all those steps with Obsea, have a look to the tutorial or the example scripts.




.. _ObsPy: https://docs.obspy.org/
.. _Inventory: https://docs.obspy.org/packages/obspy.core.inventory.html
.. _FDSN: https://docs.obspy.org/packages/obspy.clients.fdsn.html
.. _SDS: https://docs.obspy.org/master/packages/autogen/obspy.clients.filesystem.sds.html
.. _here: https://docs.obspy.org/packages/obspy.core.html#waveform-data
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable/
.. _Xarray: http://xarray.pydata.org/en/stable/