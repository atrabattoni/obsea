=========
Changelog
=========

Version 0.1
===========

- BREAKING: responses are no longer attached to streams (``Stream.attach_response``
  is deprecated in ObsPy). ``load_stream`` lost its ``inventory`` argument, and
  ``time_frequency`` gained an ``inventory`` keyword which is required when
  ``water_level`` is set to remove the instrumental response.
