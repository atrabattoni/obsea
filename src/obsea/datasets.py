"""
Dataset module.

Used to retrieve the absolut path of toys datasets shiped with obsea.

Attributes
----------
data_path : string
    directory where datasets are stored.

"""
import os
import obsea

data_path = os.path.join(os.path.dirname(obsea.__file__), 'data')


def get_dataset_path(dataset):
    """
    Give absolute path of toys datasets.

    Parameters
    ----------
    dataset : string
        Dataset name. Can be 'ais_cls', 'ais_marine_traffic', 'mmsi_list',
        'station_list' or 'ais_week_cls'.

    Returns
    -------
    string
        Absolute path.

    """
    return os.path.join(data_path, dataset + '.csv')
