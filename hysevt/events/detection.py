"""
Detection of hydro-sedimentary events.

author:
Amalie Skålevåg
skalevag2@uni-potsdam.de

"""
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import hysevt.events.metrics
import logging
from hysevt.utils.tools import log,get_freq_in_hours

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def locmin(x: np.ndarray, y: np.ndarray,window=21):
    """Get local minima.

    Transcribed into python from Anatolii Tsyplenkov's loadflux R package, `hydro_events` function: https://github.com/atsyplenkov/loadflux/blob/beb8266878342ec0757d55b0c2b4b809b6c20940/R/hydro_events.R

    Args:
        x (numpy.ndarray): streamflow values
        y (numpy.ndarray[np.datetime64]): datetime values corresponding to streamflow values
        window (int, optional): window size for local minima search, in hours. Defaults to 21.

    Returns:
        pandas.Series: timeseries with localminima indicated with boolean values

    References:
    @computer_program{Tsyplenkov2022,
        author = {Anatolii Tsyplenkov},
        doi = {10.5281/ZENODO.6992087},
        month = {8},
        title = {atsyplenkov/loadflux: Zenodo release},
        url = {https://zenodo.org/record/6992087},
        year = {2022},
    }
    @report{Sloto1996,
        author = {Ronald A Sloto and Michèle Y Crouse and Gordon P Eaton},
        doi = {10.3133/wri964040},
        journal = {Water-Resources Investigations Report},
        title = {HYSEP: A Computer Program for Streamflow Hydrograph Separation and Analysis},
        url = {https://pubs.er.usgs.gov/publication/wri964040},
        year = {1996},
    }
    """
    
    # function
    timestep = get_freq_in_hours(y) # frequency in hours
    
    # timesteps in window
    N2star = int(np.round(window/timestep))
    
    # make sure window size is odd number
    if N2star%2==0:
        N2star += 1
    
    # length of timeseries
    Nobs = len(x)
    
    # make window args
    Nfil = int((N2star - 1) / 2)
    Mid = int(N2star/2)
    
    # get local minima
    LocMin = [x[i-N2star : i].min()  == x[i- Mid] for i in range(N2star,Nobs+1)]
    
    # pad with zeroes on edges
    LocMin = np.array([False]*Nfil + LocMin + [False]*Nfil)

    return LocMin

@log(logger)
def hydro_events(qt = None, q = None, t = None, window=21):
    """Identify hydrological events with hydrograph separation using local minima method.

    Either pass a streamflow timeseries qt, or streamflow values and datetime values separately.

    Transcribed into python from Anatolii Tsyplenkov's loadflux R package, `hydro_events` function: https://github.com/atsyplenkov/loadflux/blob/beb8266878342ec0757d55b0c2b4b809b6c20940/R/hydro_events.R

    Args:
        qt (pandas.Series, optional): timeseries with streamflow values, must have DatetimeIndex or datetime like index
        q (np.ndarray, optional): streamflow values
        t (np.ndarray, optional): datetime values corresponding to streamflow values
        window (int, optional): window size for local minima search, in hours. Defaults to 21.

    Returns:
        hydro_events: pandas.Series

    References:
    @computer_program{Tsyplenkov2022,
        author = {Anatolii Tsyplenkov},
        doi = {10.5281/ZENODO.6992087},
        month = {8},
        title = {atsyplenkov/loadflux: Zenodo release},
        url = {https://zenodo.org/record/6992087},
        year = {2022},
    }
    @report{Sloto1996,
        author = {Ronald A Sloto and Michèle Y Crouse and Gordon P Eaton},
        doi = {10.3133/wri964040},
        journal = {Water-Resources Investigations Report},
        title = {HYSEP: A Computer Program for Streamflow Hydrograph Separation and Analysis},
        url = {https://pubs.er.usgs.gov/publication/wri964040},
        year = {1996},
    }
    """
    if (q is None) and (t is None) and (qt is None):
        raise TypeError("hydro_events() missing required positional argument(s): either 'qt' or 'q' and 't' are required")
    elif (q is not None) and (t is not None) and (qt is not None):
        raise TypeError("hydro_events() received too many positional argument(s): either use 'qt' or 'q' and 't'")
    elif qt is not None:
        if type(qt.index) is not pd.DatetimeIndex:
            qt.index = pd.to_datetime(qt.index)
        q = qt.values
        t = qt.index.values
    
    # get local minima mask
    LocMin = locmin(q,t,window=window)
    # convert from boolean to integer, make to Series with datetime index
    LocMin = pd.Series(LocMin.astype(int),index=t)
    # make test series
    test = LocMin.replace(0,np.nan)
    # check for events in consequtive timesteps, keep last minima 
    LocMin[test.diff(-1) == 0] = 0
    # create event index
    hydro_events = LocMin.cumsum()
    return hydro_events


def get_hydro_events_start_end(hydro_events: pd.Series) -> pd.DataFrame:
    """Get start and end timestamps of hydrological events from timeseries of hydro_events indices.

    Args:
        hydro_events (pandas.Series): series with DatetimeIndex of hydro_events indices

    Returns:
        pandas.DataFrame: "start" and "end" timestaps for each hydrological event
    """
    start = []
    end = []
    for i,event in hydro_events.groupby(hydro_events):
        start.append(event.index[0])
        end.append(event.index[-1])
    return pd.DataFrame(np.array([start,end]).T,columns=["start","end"])

@log(logger)
def filter_for_missing_data(events_list: pd.DataFrame,gauge_data: pd.DataFrame,max_missing = 0.9, column="streamflow"):
    """Filter hydrological events, removing those with too much missing data.

    Args:
        events_list (pandas.DataFrame): list of event start and end timestamps
        gauge_data (pandas.DataFrame): timeseries of streamflow data
        max_missing (float, optional): maximum fraction of missing data allowed. Defaults to 0.9.

    Returns:
        pandas.DataFrame: filtered list of event start and end timestamps
        pandas.DataFrame: list of removed events
    """

    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data) # timeseries for each event
    event_numbers = events_list.index.to_list() # index of all events
    
    events_without_data = []
    for i,event in zip(event_numbers,events_series):
        completeness = event[column].notnull().sum()/len(event)
        if completeness < max_missing:
            events_without_data.append(i)
    # save the removed events to table
    removed_events = events_list.iloc[events_without_data,:].reset_index(drop=True)
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_without_data)))].reset_index(drop=True)
    return events_list,removed_events

@log(logger)
def filter_events_without_any_sediment_data(events_list: pd.DataFrame,gauge_data: pd.DataFrame,suspended_sediment_column="suspended_sediment"):
    """Filter out events that have no suspended sediment data.

    Args:
        events_list (pandas.DataFrame): list of event start and end timestamps
        gauge_data (pandas.DataFrame): timeseries of suspended sediment data

    Returns:
        pandas.DataFrame: filtered list of event start and end timestamps
        pandas.DataFrame: list of removed events
    """
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data) # timeseries for each event
    event_numbers = events_list.index.to_list() # index of all events
    events_without_sediment = []
    for i,series in zip(event_numbers,events_series):
        if series.isna().all()[suspended_sediment_column]:
            events_without_sediment.append(i)
    # save the removed events to table
    removed_events = events_list.iloc[events_without_sediment,:].reset_index(drop=True)
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_without_sediment)))].reset_index(drop=True)

    return events_list,removed_events

@log(logger)
def filter_events_with_missing_sediment_data(events_list: pd.DataFrame, gauge_data: pd.DataFrame, SSC_threshold: float):
    """Filter out events that have missing suspended sediment values, except where the peak is above the suspended sediment threshold.

    Args:
        events_list (pandas.DataFrame): list of event start and end timestamps
        gauge_data (pandas.DataFrame): timeseries of suspended sediment data
        SSC_threshold (float): suspended sediment threshold

    Returns:
        pandas.DataFrame: filtered list of event start and end timestamps
        pandas.DataFrame: list of removed events
    """
    if SSC_threshold is None:
        SSC_threshold = np.inf
    # get event series of events which have sediment data
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data)
    event_numbers = events_list.index.to_list()
    # loop through events
    events_missing_sediment = []
    high_events_missing_sediment = []
    for i,series in zip(event_numbers,events_series):
        if series.isna().any().any():
            # high magnitude events with missing data are kept
            if series.suspended_sediment.max() >= SSC_threshold:
                high_events_missing_sediment.append(i)
            # the rest is removed
            else:
                events_missing_sediment.append(i)
    # edit start and end dates for events missing data, but of high magnitude
    for i in high_events_missing_sediment:
        events_list.loc[i,"start"] = events_series[i].dropna().index[0]
        events_list.loc[i,"end"] = events_series[i].dropna().index[-1]
    # save the removed events to table
    removed_events = events_list.iloc[events_missing_sediment,:].reset_index(drop=True)
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_missing_sediment)))].reset_index(drop=True)

    return events_list,removed_events

@log(logger)
def filter_out_small_hydro_sediment_events(events_list: pd.DataFrame, gauge_data: pd.DataFrame, SSC_threshold: float):
    """Filter out events where the suspended sediment peak is below the suspended sediment threshold.

    Args:
        events_list (pandas.DataFrame): list of event start and end timestamps
        gauge_data (pandas.DataFrame): timeseries of suspended sediment data
        SSC_threshold (float): suspended sediment threshold

    Returns:
        pandas.DataFrame: filtered list of event start and end timestamps
        pandas.DataFrame: list of removed events
    """
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data)
    event_numbers = events_list.index.to_list()
    # loop through events again
    big_events = []
    for i,series in zip(event_numbers,events_series):
        if series.suspended_sediment.max() >= SSC_threshold:
            # events with magnitude higher then threshold are kept
            big_events.append(i)
            
    removed_events = events_list[np.logical_not(events_list.index.isin(np.array(big_events)))].reset_index(drop=True)
    # select only events with sediment data and reset index
    events_list = events_list.iloc[big_events,:].reset_index(drop=True)

    return events_list,removed_events


@log(logger)
def hydro_sediment_events(path_gauge_data: Path,
                          output_directory: Path,
                          event_station_id: str,
                          event_version_id = "AUT",
                          ma_window = "3H",
                          he_window=21,
                          SSC_threshold_by_quantile = 0.9,
                          keep_ma_gauge_data = False,
                          keep_removed_events = False,
                          SSC_threshold = None,
                          max_Q_missing = None,
                          max_SSC_missing = None) -> pd.DataFrame:
    """Run full hydro-sediment event detection.

    Loads timeseries of streamflow and suspeded sediment. 
    Identifies hydrological events with local minima hydrograph separation. 
    Then filters out events based on various criteria.
    Each action is documented in the "detection.log" file, which is saved to the same location as the output.
    The final list of hydro-sediment events are both returned by the function and saved to csv-file at 'output_file'.

    Args:
        path_gauge_data (pathlib.Path): path to csv-file containing suspended sediment and streamflow data. Index must be datetime, and must have the column headers "suspended_sediment" and "streamflow".
        output_file (pathlib.Path): path to csv-file where final hydro-sediment events list should be saved.
        event_station_id (str): abbreviation of station name or id, will be used to create the event id
        event_version_id (str, optional): abbreviation of event detection version, will be used to create the event id. Defaults to "AUT".
        ma_window (str, optional): size of median moving average window, must conform to pandas.Timedelta units. Defaults to "3H".
        he_window (int, optional): size of local minima window for hydrograph separation. Defaults to 21.
        SSC_threshold_by_quantile (float, optional): quantile used to set peak suspended sediment threshold, will be ignored if 'SSC_threshold' argument is used. If both 'SSC_threshold_by_quantile' and 'SSC_threshold' is set to None, no events will be removed due to small SSC magnitude. Defaults to 0.9.
        keep_ma_gauge_data (bool, optional): if True, the smoothed gauge data is saved to file in the same directory as 'path_gauge_data'. Defaults to False.
        keep_removed_events (bool, optional): if True, the dataframes of removed events from each filtering is saved to csv-file at same location as output. Defaults to False.
        SSC_threshold (float, optional): set peak suspended sediment threshold, will override 'SSC_threshold_by_quantile' argument. If both 'SSC_threshold_by_quantile' and 'SSC_threshold' is set to None, no events will be removed due to small SSC magnitude. Defaults to None.
        max_Q_missing (float, optional): maximum fraction of missing streamflow data allowed in each event, if too much data is missing in an event it will be removed. Defaults to None.
        max_SSC_missing (float, optional): maximum fraction of missing streamflow data allowed in each event, if too much data is missing in an event it will be removed. Defaults to None.

    Returns:
        pd.DataFrame: list of hydro-sediment events start and end timestamps, same dataframe is also saved as a csv-file at 'output_directory'
    """
    output_file = output_directory.joinpath("hydro_sediment_events_list.csv")
    
    # create local log
    locallog = logging.FileHandler(filename=output_directory.joinpath(f'{__name__.split(".")[-1]}.log'),mode='w')
    locallog.setLevel(logging.INFO)
    logger.addHandler(locallog)
    
    logger.info(f"############ PREAMBLE ############")
    logger.info(f"Gauge data file: {path_gauge_data}")
    # get gauging station data
    gauge_data = hysevt.events.metrics.get_gauge_data(path_gauge_data)
    logger.info(f"Gauge data successfully loaded.")
    logger.info(f"Gauge data temporal extent: {gauge_data.index[0]}-{gauge_data.index[-1]}")
    # document the location of output and filenames
    logger.info(f"Output directory: {output_directory}")
    logger.info(f"Final output file: {output_file.name}")
    # hydrograph separation settings
    logger.info(f"Local minima window: {he_window} hours")
    logger.info(f"Rolling median window: {ma_window}")
    if max_Q_missing is not None:
        logger.info(f"Hydrological events must have more than {max_Q_missing*100}% valid streamflow data.")
    logger.info(f"filter_SSC_quantile = {SSC_threshold_by_quantile}")
    if max_SSC_missing is not None:
        logger.info(f"Hydro-sediment events must have more than {max_SSC_missing*100}% valid suspended sediment data.")
    
    # set peak SSC threshold
    if SSC_threshold is None and SSC_threshold_by_quantile is not None:
        # get SSC threshold from quantile
        SSC_threshold = gauge_data.suspended_sediment.quantile(SSC_threshold_by_quantile)
        logger.info(f"{SSC_threshold_by_quantile}-quantile SSC threshold = {SSC_threshold}")
    elif SSC_threshold is not None:
        logger.info(f"User-defined SSC threshold = {SSC_threshold}")
    else:
        SSC_threshold = None

    logger.info(f"############ DETECTION ############")
    # apply moving median smoothing
    gauge_data_smooth = gauge_data.rolling(ma_window,center=True).median()
    logger.info(f"Applied {ma_window}-median smoothing to gauge data.")
    
    if keep_ma_gauge_data:
        # loc to save smooth data
        path_gauge_data_smooth = path_gauge_data.parent.joinpath(f"{path_gauge_data.name.split('.')[0]}_{ma_window}-median_smooth.csv")
        logger.info(f"Keeping {ma_window}-median smoothed gauge data: {path_gauge_data_smooth}")
        gauge_data_smooth.to_csv(path_gauge_data_smooth)
        logger.info(f"Saving {ma_window}-median smoothed gauge data to file: {path_gauge_data_smooth}")
    
    
    # make temp filename
    temp_output_file = output_directory.joinpath("all_hydro_events.csv")
    
    # get hydro events
    logger.info(f"Detecting hydrological events ...")
    gauge_data["he"] = hysevt.events.detection.hydro_events(qt=gauge_data_smooth.streamflow,window=he_window)
    events_list = get_hydro_events_start_end(gauge_data.he)
    logger.info(f"{len(events_list)} hydrological events detected.")
    
    # remove hydro events with too much missing data
    if max_Q_missing is not None:
        events_list,removed_events = filter_for_missing_data(events_list,gauge_data=gauge_data,max_missing=max_Q_missing,column="streamflow")
        logger.info(f"{len(removed_events)} hydrological events removed due too much missing data (>{max_Q_missing*100}%).")
        if keep_removed_events:
            removed_events.to_csv(output_file.parent.joinpath("events_with_too_much_missing_streamflow.csv"),index=False)
        logger.info(f"{len(events_list)} events left.")
    
    # save hydrological events to file
    events_list.to_csv(temp_output_file)
    logger.info(f"Hydrological events saved to file: {temp_output_file}")
    
    # filter out events without sediment data
    events_list,removed_events = filter_events_without_any_sediment_data(events_list,gauge_data=gauge_data)
    logger.info(f"{len(removed_events)}, events were removed due to no sediment data.")
    if keep_removed_events:
        removed_events.to_csv(output_file.parent.joinpath("events_without_sediment_data.csv"),index=False)
    logger.info(f"{len(events_list)} events left.")
    
    # remove events with too much missing suspended sediment data
    if max_SSC_missing is not None:
        events_list,removed_events = filter_for_missing_data(events_list,gauge_data=gauge_data,max_missing=max_SSC_missing,column="suspended_sediment")
        logger.info(f"{len(removed_events)} hydro-sediment events removed due too much missing SSC data (>{max_SSC_missing*100}%).")
        if keep_removed_events:
            removed_events.to_csv(output_file.parent.joinpath("events_with_too_much_missing_SSC.csv"),index=False)
        logger.info(f"{len(events_list)} events left.")

    # filter out events with missing sediment data, unless they are above SSC threshold
    events_list,removed_events = filter_events_with_missing_sediment_data(events_list,gauge_data=gauge_data,SSC_threshold=SSC_threshold)
    logger.info(f"{len(removed_events)} events were removed due to incomplete sediment data, except where the peak was above SSC_threshold")
    if keep_removed_events:
        # save the removed events to table
        removed_events.to_csv(output_file.parent.joinpath("events_missing_sediment_data.csv"),index=False)
    logger.info(f"{len(events_list)} events left.")

    if SSC_threshold is not None:
        # filter out events below SSC threshold, i.e. small events
        events_list,removed_events = filter_out_small_hydro_sediment_events(events_list,gauge_data=gauge_data,SSC_threshold=SSC_threshold)
        logger.info(f"{len(removed_events)} events were removed due small SSC maginitude.")
        if keep_removed_events:
            removed_events.to_csv(output_file.parent.joinpath("events_small_magnitude.csv"), index=False)
        logger.info(f"{len(events_list)} events left.")
    
    # add event id numbers 
    event_id_numbers = []
    for y,sub in events_list.groupby([time.year for time in events_list.start]):
        event_id_numbers = event_id_numbers + [f"{event_station_id}-{event_version_id}-{y}-{i+1:03d}" for i in sub.reset_index(drop=True).index]
    assert len(events_list) == len(event_id_numbers)
    events_list.insert(0,"event_id",event_id_numbers)
    events_list = events_list.set_index("event_id")
    
    # save to file
    events_list.to_csv(output_file,index=False)
    logger.info(f"Final event list saved at {output_file}")
    
    return events_list

@log(logger)
def call_local_minimum_script(path_gauge_data: Path,output_file: Path, path_rscript=Path(__file__).parent.joinpath("identify_events_local_minimum.R")):
    """Identifies hydrological events from streamflow data using the local minima method.

    Calls on an R-script that uses the `hydro_events` function from the `loadflux` package (https://github.com/atsyplenkov/loadflux.git).

    Args:
        path_gauge_data (pathlib.Path): absolute path to csv file containing gaunging station data
        output_file (pathlib.Path): path to csv file where output is to be saved
        path_rscript (pathlib.Path, optional): path to R script running the local minimum method. Defaults to Path(__file__).parent.joinpath("identify_events_local_minimum.R").
    """
    # check that files exist
    if not path_gauge_data.is_file():
        raise FileNotFoundError(f"{path_gauge_data}")
    elif not path_rscript.is_file():
        raise FileNotFoundError(f"{path_rscript}")

    # create command
    cmd = ["Rscript", f"{path_rscript}", f"{path_gauge_data}", f"{output_file}"]
    # run process
    logger.info(f"Calling R-script with CMD : {' '.join(cmd)}")
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logger.info(out.stdout)

@log(logger)
def local_minimum(path_gauge_data: Path,output_file: Path, event_station_id: str, event_version_id = "AUT", ma_window = "3H", filter_SSC_quantile=0.9, keep_ma_gauge_data = False, path_rscript=Path(__file__).parent.joinpath("identify_events_local_minimum.R")) -> pd.DataFrame:
    """Identifies hydrological events from streamflow data using the local minima method. 
    Then filters out events where suspended sediment data is missing or incomplete.

    Args:
        path_gauge_data (pathlib.Path): absolute path to csv file containing gaunging station data
        event_station_id (str): the guaging station id.
        event_version_id (str): the event detection method used. Defaults to "AUT".
        output_file (pathlib.Path): path to csv file where output is to be saved

    Returns:
        pd.DataFrame: list of event start and end timestamps

    References:
    @computer_program{Tsyplenkov2022,
        author = {Anatolii Tsyplenkov},
        doi = {10.5281/ZENODO.6992087},
        month = {8},
        title = {atsyplenkov/loadflux: Zenodo release},
        url = {https://zenodo.org/record/6992087},
        year = {2022},
    }
    @report{Sloto1996,
        author = {Ronald A Sloto and Michèle Y Crouse and Gordon P Eaton},
        doi = {10.3133/wri964040},
        journal = {Water-Resources Investigations Report},
        title = {HYSEP: A Computer Program for Streamflow Hydrograph Separation and Analysis},
        url = {https://pubs.er.usgs.gov/publication/wri964040},
        year = {1996},
    }

    """
    # create local log
    locallog = logging.FileHandler(filename=output_file.parent.joinpath(f'{__name__.split(".")[-1]}.log'),mode='w')
    locallog.setLevel(logging.INFO)
    logger.addHandler(locallog)
    
    logger.info(f"Gauge data file: {path_gauge_data}")
    # get gaugin station data
    gauge_data = hysevt.events.metrics.get_gauge_data(path_gauge_data)
    logger.info(f"Gauge data successfully loaded.")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Rolling median window: {ma_window}")
    logger.info(f"filter_SSC_quantile = {filter_SSC_quantile}")
    # get SSC threshold
    SSC_threshold = gauge_data.suspended_sediment.quantile(filter_SSC_quantile)
    logger.info(f"{filter_SSC_quantile}-quantile SSC threshold = {SSC_threshold}")
    if keep_ma_gauge_data:
        logger.info(f"Keeping {ma_window}-median smoothed gauge data.")


    # apply moving median smoothing
    gauge_data_smooth = gauge_data.rolling(ma_window,center=True).median()
    logger.info(f"Applied {ma_window}-median smoothing to gauge data.")

    # loc to save smooth data
    path_gauge_data_smooth = path_gauge_data.parent.joinpath(f"{path_gauge_data.name.split('.')[0]}_{ma_window}-median_smooth.csv")
    gauge_data_smooth.to_csv(path_gauge_data_smooth)
    logger.info(f"Saving {ma_window}-median smoothed gauge data to file: {path_gauge_data_smooth}")
    
    # make temp filename
    temp_output_file = Path(str(output_file).split(".")[0]+"_all_hydro_events.csv")
    call_local_minimum_script(path_gauge_data_smooth,temp_output_file,path_rscript=path_rscript)
    
    if not keep_ma_gauge_data:
        # remove smoothed gauge data 
        path_gauge_data_smooth.unlink()
        logger.info(f"Deleted {ma_window}-median smoothed gauge data file.")
    
    # import list of hydro events
    events_list = pd.read_csv(temp_output_file)
    logger.info(f"{len(events_list)} hydrological events detected.")
    
    # filter out events without sediment data
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data) # timeseries for each event
    event_numbers = events_list.index.to_list() # index of all events
    events_without_sediment = []
    for i,series in zip(event_numbers,events_series):
        if series.isna().all().suspended_sediment:
            events_without_sediment.append(i)
    # save the removed events to table
    events_list.iloc[events_without_sediment,:].reset_index(drop=True).to_csv(output_file.parent.joinpath("events_without_sediment_data.csv"),index=False)
    logger.info(f"{len(events_without_sediment)}, events were removed due to no sediment data.")
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_without_sediment)))].reset_index(drop=True)
    logger.info(f"{len(events_list)} events left.")

    # get event series of events which have sediment data
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data)
    event_numbers = events_list.index.to_list()
    # loop through events again
    events_missing_sediment = []
    high_events_missing_sediment = []
    for i,series in zip(event_numbers,events_series):
        if series.isna().any().any():
            # high magnitude events with missing data are kept
            if series.suspended_sediment.max() >= SSC_threshold:
                high_events_missing_sediment.append(i)
            # the rest is removed
            else:
                events_missing_sediment.append(i)
    # edit start and end dates for events missing data, but of high magnitude
    for i in high_events_missing_sediment:
        events_list.loc[i,"start"] = events_series[i].dropna().index[0]
        events_list.loc[i,"end"] = events_series[i].dropna().index[-1]
        
    # save the removed events to table
    events_list.iloc[events_missing_sediment,:].reset_index(drop=True).to_csv(output_file.parent.joinpath("events_missing_sediment_data.csv"),index=False)
    logger.info(f"{len(events_missing_sediment)} events were removed due to missing sediment data.")
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_missing_sediment)))].reset_index(drop=True)
    logger.info(f"{len(events_list)} events left.")
    

    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data)
    event_numbers = events_list.index.to_list()
    # loop through events again
    big_events = []
    for i,series in zip(event_numbers,events_series):
        if series.suspended_sediment.max() >= SSC_threshold:
            # events with magnitude higher then threshold are kept
            big_events.append(i)
            
    events_list[np.logical_not(events_list.index.isin(np.array(big_events)))].reset_index(
        drop=True
    ).to_csv(output_file.parent.joinpath("events_missing_sediment_data.csv"), index=False)
    logger.info(f"{len(events_list)-len(big_events)} events were removed due small SSC maginitude.")
    # select only events with sediment data and reset index
    events_list = events_list.iloc[big_events,:].reset_index(drop=True)
    logger.info(f"{len(events_list)} events left.")
    

    # convert columns to datetime
    events_list.start = pd.to_datetime(events_list.start)
    events_list.end = pd.to_datetime(events_list.end)
    
    # add event id numbers 
    event_id_numbers = []
    for y,sub in events_list.groupby([time.year for time in events_list.start]):
        event_id_numbers = event_id_numbers + [f"{event_station_id}-{event_version_id}-{y}-{i+1:03d}" for i in sub.reset_index(drop=True).index]
        
    assert len(events_list) == len(event_id_numbers)
    events_list.insert(0,"event_id",event_id_numbers)

    # save to file
    events_list.to_csv(output_file,index=False)
    logger.info(f"Final event list saved at {output_file}")

    return events_list

####### MAIN FUNCTION TO RUN SCRIPT FROM TERMINAL #######
def main(file_gauge_data,file_events,file_annual_streamflow,file_annual_sediment,output_file):
    
    # create local log
    locallog = logging.FileHandler(filename=output_file.parent.joinpath(f'{__name__.split(".")[-1]}.log'),mode='w')
    locallog.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    locallog.setFormatter(formatter)
    logger.addHandler(locallog)
    
    logger.info(f"Gauge data file: {file_gauge_data}")
    logger.info(f"Output file: {output_file}")
    # import data
    data = hysevt.events.metrics.get_gauge_data(file_gauge_data)
    events_list = pd.read_csv(file_events)
    annual_sediment_yield = pd.read_csv(file_annual_sediment,index_col=0)
    annual_streamflow_volume = pd.read_csv(file_annual_streamflow,index_col=0)
    # calucalte the metrics
    #event_metrics = calculate_event_metrics(events_list,data,annual_sediment_yield,annual_streamflow_volume)
    
    #hysteresis_index = calc_hysteresis_index(file_gauge_data,file_events)
    #event_metrics = event_metrics.join(hysteresis_index.drop(columns=["start","end"]))
    event_metrics = event_metrics.drop(columns="X")
    
    # save the metrics to file
    event_metrics.to_csv(output_file,index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--gaugefile', dest="file_gauge_data", type=Path, required=True, help='guage data file with timeseries data from gauging station')
    parser.add_argument('-e', '--eventfile', dest="file_event_list", type=Path, required=True, help='event list file with start and end times for events in columns "start" and "end" respectively')
    parser.add_argument('-s', '--annualsediment', dest="file_annual_sediment", type=Path, required=True, help='timeseries of annual suspended sediment yield from gauging station')
    parser.add_argument('-q', '--annualstreamflow', dest="file_annual_streamflow", type=Path, required=True, help='timeseries of annual streamflow volume from gauging station')
    parser.add_argument('-o', '--outputfile', dest="output_file", type=Path, required=True, help='output file')
    args = parser.parse_args()

    main(file_gauge_data=args.file_gauge_data,file_events=args.file_event_list,file_annual_streamflow=args.file_annual_streamflow,file_annual_sediment=args.file_annual_sediment,output_file=args.output_file)