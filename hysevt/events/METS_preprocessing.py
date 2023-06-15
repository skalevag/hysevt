"""
Preprocessing for Multivariate Event Time Series analysis (METS) clustering after Javed et al., 2020 (https://doi.org/10.1016/j.jhydrol.2020.125802).

Author: Amalie Skålevåg
Email: skalevag2@uni-potsdam.de
"""

import scipy
import numpy as np
import pandas as pd
import hysevt.events.metrics
import hysevt.utils.tools
from pathlib import Path

def standard_event_length(x: np.ndarray, n=50, k=3, s=0) -> np.ndarray:
    """Standardises an event timeseries to a specified number of timesteps univariate spline fitting (scipy.interpolate.UnivariateSpline).

    Args:
        x (np.ndarray): timeseries
        n (int, optional): number of timesteps in output. Defaults to 50.
        k (int, optional): degree of the smoothing spline, see scipy.interpolate.UnivariateSpline for details. Defaults to 3.
        s (int, optional): smoothing factor used to choose the number of knots, see scipy.interpolate.UnivariateSpline for details. Defaults to 0.

    Returns:
        np.ndarray: timeseries standardised to n timesteps
    """
    return scipy.interpolate.UnivariateSpline(np.arange(len(x)), x, k=k, s=s)(
        np.linspace(0, len(x), n)
    )

def normalise_magnitude(x):
    """Standardises the magnitude of events to (0,1).
    """
    return (x - x.min()) / (x.max() - x.min())

def preprocess_timeseries(
    x: np.ndarray, filter_window_length=9, filter_polyorder=4, n=50, return_full_results=False
):
    """Preprocess timeseries after apadted procedure from Javed et al., 2020.

    Args:
        x (numpy.ndarray): timeseries to be preprocessed
        filter_window_length (int, optional): length of the filter window for Savitzky-Golay filter, see `scipy.signal.savgol_filter` for details. Defaults to 9.
        filter_polyorder (int, optional): order of the polynomial used in Savitzky-Golay filter, see `scipy.signal.savgol_filter` for details. Defaults to 4.
        n (int, optional): number of timesteps (standardised event length). Defaults to 50.
        return_full_results (bool, optional): whether to return all steps in the preprocessing, i.e. also the smoothed and standardised event lengths. Defaults to False.

    Returns:
        numpy.ndarray: _description_
    """
    x_smooth = scipy.signal.savgol_filter(x, filter_window_length, filter_polyorder,mode="nearest")
    x_stand = standard_event_length(x_smooth, n=n)
    x_norm = normalise_magnitude(x_stand)
    if return_full_results:
        return x_smooth, x_stand, x_norm
    else:
        return x_norm


def preprocess_event(
    event_data: pd.DataFrame, filter_window_length=9, filter_polyorder=3, n=50,
):
    """Preprocesses an event timeseries (multivariate) after Javed et al., 2020.

    Args:
        event_data (pandas.DataFrame): dataframe of event data, with timeseries index and each variable in a column
        filter_window_length (int, optional): length of the filter window for Savitzky-Golay filter, see `scipy.signal.savgol_filter` for details. Defaults to 9.
        filter_polyorder (int, optional): order of the polynomial used in Savitzky-Golay filter, see `scipy.signal.savgol_filter` for details. Defaults to 4.
        n (int, optional): number of timesteps (standardised event length). Defaults to 50.

    Returns:
        pandas.DataFrame: preprocessed event
    """
    event_data_preprocessed = pd.DataFrame(
        np.array(
            [
                preprocess_timeseries(
                    event_data[var].values,
                    filter_window_length=filter_window_length,
                    filter_polyorder=filter_polyorder,
                    n=n,
                    return_full_results=False,
                )
                for var in event_data.columns
            ]
        ).T,
        columns=event_data.columns,
    )
    return event_data_preprocessed


def preprocess_all_events(unprocessed_events: list, filter_window_length=9, filter_polyorder=3, n=50) -> np.ndarray:
    """Preprocesses a list of multivariate timeseries with y variables from x events.
    Returns a 3-dimensional numpy array of the shape (number of events, number of variables in multivariate timeseries, n).

    Args:
        unprocessed_events (list): list of multivariate timeseries
        filter_window_length (int, optional): length of the filter window for Savitzky-Golay filter, see `scipy.signal.savgol_filter` for details. Defaults to 9.
        filter_polyorder (int, optional): order of the polynomial used in Savitzky-Golay filter, see `scipy.signal.savgol_filter` for details. Defaults to 4.
        n (int, optional): number of timesteps (standardised event length). Defaults to 50.

    Returns:
        numpy.ndarray: array of shape (x,y,n)
    """
    # preprocess all events in list
    preprocessed_events = [preprocess_event(event_data,filter_window_length=filter_window_length, filter_polyorder=filter_polyorder, n=n,).values.T for event_data in unprocessed_events]
    # return stack of events
    return np.stack(preprocessed_events)



def main(file_gauge_data: Path, file_event_list: Path, output_filehead=None):

    # import dataframe with event start and end timestamps
    event_list = pd.read_csv(file_event_list,index_col=0)

    # save the event ids to list
    hysevt.utils.tools.save_list_to_txt(event_list.index.to_list(),file_event_list.parent.joinpath(f"{output_filehead}_event_id_list.txt"))

    # data
    gauge_data = hysevt.events.metrics.get_gauge_data(file_gauge_data)

    # extract event series from all events to list
    unprocessed_events = [hysevt.events.metrics.get_event_data(event.start, event.end, gauge_data).dropna() for i, event in event_list.iterrows()]
    # preprocess data for all events and stack to array
    preprocessed_data = preprocess_all_events(unprocessed_events)

    # check that preprocessed data do not contain NaNs
    assert(np.logical_not(np.isnan(preprocessed_data).any()))

    # save to file
    np.save(
        file_event_list.parent.joinpath(
            f"{output_filehead}_preprocessed_events_for_METS_clustering.npy"
        ),
        preprocessed_data,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--gaugefile', dest="file_gauge_data", type=Path, required=True, help='guage data file with timeseries data from gauging station')
    parser.add_argument('-e', '--eventfile', dest="file_event_list", type=Path, required=True, help='event list file with start and end times for events in columns "start" and "end" respectively')
    parser.add_argument('-o', '--outputname', dest="output_filehead", type=str, default="my", help='filehead of output, "preprocessed_events_for_METS_clustering.npy" will be appended, saved to same location as event list file')
    args = parser.parse_args()

    main(file_gauge_data=args.file_gauge_data, file_event_list=args.file_event_list,output_filehead=args.output_filehead)