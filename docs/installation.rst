============
Installation
============

We highly recommend to use ``conda`` to install python packages. It is shipped with Anaconda_ (or Miniconda_ for a smaller footprint), a scientific Python distribution which allows to very quickly configure a Python scientific setup. Unfortunately, the package is presently only available on ``pip`` and cannot directly be installed with ``conda``. We recommend to install dependencies with ``conda`` and then install the package using ``pip``.

As several required packages to install obsea can only be found on the ``conda-forge`` channel, we highly recommend to work on a separate environment which only contains packages downloaded from the ``conda-forge`` channel. Especially working with the ``base`` environment it not recommended as it is known that mixing packages from both channels is a source of bugs_. Two scenario cases are possible: installing the package and its dependencies from scratch into a new environment (start at 1.) or installing those in a already existing environment which was setup to use the ``conda-forge`` channel (start at 4.).

1. Install Anaconda_ or Miniconda_ following the instructions on their website.
2. Create a new environment and activate it (choose the environment name you want, here ``obsea`` is chosen)::

    conda create -n obsea  
    conda activate obsea

3. Use ``conda-forge`` as the priority channel an remove the ``defaults`` channel::

    conda config --env --add channels conda-forge

4. Install dependencies::

    conda install cartopy colorcet dask matplotlib numba numpy obspy pandas python scikit-learn scipy shapely xarray netcdf4 pyproj

5. Install obsea::

    pip install obsea

Congratulation you just installed obsea ! Remember to activate the environment where you installed obsea each time you need to use it.

.. _Anaconda: https://www.anaconda.com/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _bugs: https://conda-forge.org/docs/user/tipsandtricks.html
