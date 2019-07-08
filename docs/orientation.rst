Orientation script
------------------

This script allows to retrieve the orientation of the RR03 station of the RHUM-RUM experiment. It returns the most probable azimuth and uncertainties in a CSV file. The Inputs and Parameter parts can be modified to fulfill the user needs.

.. literalinclude:: ../examples/orientation.py
   :language: python

.. highlight:: none

Output::

   ------------------
   ORIENTATION SCRIPT
   ------------------
   Process AIS.
   Process Data.
   [########################################] | 100% Completed | 35.4s
   Estimate Uncertainties.
   [########################################] | 100% Completed |  2.2s
   orientation for 1 routes = 76.8° +/- 4.7°
   orientation for 53 routes = 76.8° +/- 0.7°
