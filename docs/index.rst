=====
Obsea
=====

Obsea metamorphoses Ocean-Bottom Seismometers (OBS) into watchful observers of the Oceans. Sensing infrasound, OBSs can record marine mammal vocalizes, ship traffic noise, iceberg activity and more at very long ranges (~100 km). This package aims at automatically recognizing and localizing acoustical sources that can be recorded by OBSs.

This package allows to calibrate OBSs with ship noise i.e. to properly reorient and re-localize it on the ocean floor. To that purpose, noise of ships passing close by OBSs is analyzed and compared with known ship trajectories obtained from the Automatic Identification System. More details can be found in the following, peer-reviewd paper. If you find obsea useful, please consider citing it.

    Trabattoni, A., Barruol, G., Dreo, R., Boudraa, A.O. & Fontaine, F.R. (2020) Orienting and locating ocean-bottom seismometers from ship noise analysis. Geophys J Int, 220, 1774–1790. `doi:10.1093/gji/ggz519 <https://doi.org/10.1093/gji/ggz519>`_

This package also allows to detect, localize and track ships and potentially whales. More details can be found in the following, peer-reviewd paper. If you find obsea useful, please consider citing it.

    Trabattoni, A., Barruol, G., Dréo, R., & Boudraa, A. (2023). Ship detection and tracking from single ocean-bottom seismic and hydroacoustic stations. The Journal of the Acoustical Society of America, 153(1), 260–273. `doi:10.1121/10.00168109 <https://doi.org/10.1121/10.00168109>`_

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
