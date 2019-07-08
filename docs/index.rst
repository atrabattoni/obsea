==========
Obsea
==========

Obsea metamorphoses Ocean-Bottom Seismometers (OBS) into watchful observers of the Oceans. Sensing infrasound, OBSs can record marine mammal vocalizes, ship traffic noise, iceberg activity and more at very long ranges (~100 km). This package aims at automatically recognizing and localizing acoustical sources that can be recorded by OBSs.

This package is still on heavy development. Presently, it gathers the work of an ongoing PhD thesis and only allows to calibrate the OBS i.e. to properly reorient and re-localize it on the ocean floor. To that purpose, noise of ships passing close by OBSs is analyzed and compared with known ship trajectories obtained from the Automatic Identification System.  

A small toy dataset of AIS data is provided to test the package. It consists of a few weeks of data around the location of a station (RR03) deployed during the `RHUM-RUM`__ experiment from two different providers. Associated seismological data can be downloaded from the RESIF_ web services.

Obsea is provided under the `Version 3 of the GNU Lesser General Public License`__. 


Contents
========

.. toctree::
   :maxdepth: 1

   Installation <installation>
   Workflow <workflow>
   Tutorial <tutorial>
   Scripts Examples <examples>
   Module Reference <api/modules>
   Changelog <changelog>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _RR: http://www.rhum-rum.net/en/
__ RR_
.. _RESIF: https://www.resif.fr/
.. _LGPL: https://www.gnu.org/licenses/lgpl.txt
__ LGPL_
