=========
Changelog
=========

Version 0.1 (2026-07-20)
========================

- BREAKING: responses are no longer attached to streams (``Stream.attach_response``
  is deprecated in ObsPy). ``load_stream`` lost its ``inventory`` argument, and
  ``time_frequency`` gained an ``inventory`` keyword which is required when
  ``water_level`` is set to remove the instrumental response.
- Examples and docs: FDSN client switched from RESIF to EPOSFR, Read the Docs
  configuration added, installation guide simplified to a plain
  ``pip install obsea``.
- Packaging: migrate from ``setup.py``/``setup.cfg`` to ``pyproject.toml`` with
  proper dependency declaration (#2), add the missing ``netCDF4`` dependency,
  and use ``importlib.metadata`` instead of the deprecated ``pkg_resources``
  to expose ``__version__``.
- FIX: POSIX time conversions in the ``gis`` module used an epoch offset by one
  second; tracks now round-trip correctly between datetimes and LineStrings.
- FIX: allow ``datetime64`` time coordinates in beamforming and cepstral
  localization.
- FIX: compatibility with newer Shapely (deprecated usages removed), SciPy
  (``scipy.signal.windows`` import) and pandas (``read_csv`` ``squeeze``
  argument removal, in the examples).

Version 0.0.4 (2023-01-18)
==========================

- Add the ``detection`` module: tonal and impulsive (cepstral) detection.
- Add the ``tracking`` module: batch SVD filtering, marginals and peak
  extraction, tracker.
- Add the ``raytracing`` module: TDOA and PDOA computation.
- ``gis``: use datetime coordinates instead of floats and DataArrays instead
  of LineStrings for tracks, switch to pyproj, add ``get_cpa``, keep track
  attributes through processing.
- ``ais``: keep static ship information with the CLS notation, clean provided
  datasets and add a one-week CLS toy dataset.
- ``azigram``: extract the ``intensity`` function, allow azimuthal stability
  computation along frequency, add the ``iid`` option.
- ``svd_filter``: optional mean removal.
- Documentation improvements (channels description, detection module) and
  compatibility fixes (Shapely and pandas deprecations).

Version 0.0.3 (2020-01-14)
==========================

- ``orientation_frequency`` accepts tracks as xarray objects (``track2xarr``).
- Add acoustical center correction to beamforming.
- Remove the ``blur`` function.
- Update the localization example and documentation (article citation).

Version 0.0.2 (2019-07-10)
==========================

Initial public release with the ``ais``, ``beamforming``, ``core``,
``datasets``, ``gis``, ``io``, ``plot``, ``processing`` and ``station``
modules, together with toy AIS datasets.
