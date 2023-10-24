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

@log(logger)
def locmin(x: np.ndarray, y: np.ndarray[np.datetime64],window=21):
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
    if N2star==0:
        N2star =+ 1
    
    # length of timeseries
    Nobs = len(x)
    
    # make window args
    Ngrp = np.ceil(Nobs/N2star)
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

@log(logger)
def get_hydro_events_start_end(hydro_events):
    start = []
    end = []
    for i,event in hydro_events.groupby(hydro_events):
        start.append(event.index[0])
        end.append(event.index[-1])
    return pd.DataFrame(np.array([start,end]).T,columns=["start","end"])

@log(logger)
def filter_hydro_events(events_list,gauge_data,max_missing = 0.9):
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data) # timeseries for each event
    event_numbers = events_list.index.to_list() # index of all events
    
    events_without_data = []
    for i,event in zip(event_numbers,events_series):
        completeness = event.streamflow.notnull().sum()/len(event)
        if completeness < max_missing:
            events_without_data.append(i)
    # save the removed events to table
    removed_events = events_list.iloc[events_without_data,:].reset_index(drop=True)
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_without_data)))].reset_index(drop=True)
    return events_list,removed_events

@log(logger)
def filter_events_without_sediment_data(events_list,gauge_data):
    events_series = hysevt.events.metrics.get_event_series(events_list,gauge_data) # timeseries for each event
    event_numbers = events_list.index.to_list() # index of all events
    events_without_sediment = []
    for i,series in zip(event_numbers,events_series):
        if series.isna().all().suspended_sediment:
            events_without_sediment.append(i)
    # save the removed events to table
    removed_events = events_list.iloc[events_without_sediment,:].reset_index(drop=True)
    # select only events with sediment data and reset index
    events_list = events_list[np.logical_not(events_list.index.isin(np.array(events_without_sediment)))].reset_index(drop=True)

    return events_list,removed_events

@log(logger)
def filter_events_with_missing_sediment_data(events_list,gauge_data,SSC_threshold):
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
def filter_out_small_hydro_sediment_events(events_list,gauge_data,SSC_threshold):
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
def hydro_sediment_events(path_gauge_data,
                          output_file,
                          event_station_id: str,
                          event_version_id = "AUT",
                          ma_window = "3H",
                          he_window=21,
                          filter_SSC_quantile = 0.9,
                          keep_ma_gauge_data = False,
                          keep_removed_events = False,
                          SSC_threshold = None):
    # create local log
    locallog = logging.FileHandler(filename=output_file.parent.joinpath(f'{__name__.split(".")[-1]}.log'),mode='w')
    locallog.setLevel(logging.INFO)
    logger.addHandler(locallog)
    
    logger.info(f"############ PREAMBLE ############")
    logger.info(f"Gauge data file: {path_gauge_data}")
    # get gaugin station data
    gauge_data = hysevt.events.metrics.get_gauge_data(path_gauge_data)
    logger.info(f"Gauge data successfully loaded.")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Local minima window: {he_window} hours")
    logger.info(f"Rolling median window: {ma_window}")
    logger.info(f"filter_SSC_quantile = {filter_SSC_quantile}")
    
    # set peak SSC threshold
    if SSC_threshold is None:
        # get SSC threshold from quantile
        SSC_threshold = gauge_data.suspended_sediment.quantile(filter_SSC_quantile)
        logger.info(f"{filter_SSC_quantile}-quantile SSC threshold = {SSC_threshold}")
    else:
        logger.info(f"User-defined SSC threshold = {SSC_threshold}")

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
    temp_output_file = Path(str(output_file).split(".")[0]+"_all_hydro_events.csv")
    
    logger.info(f"############ DETECTION ############")
    # get hydro events
    gauge_data["he"] = hysevt.events.detection.hydro_events(qt=gauge_data_smooth.streamflow,window=he_window)
    events_list = get_hydro_events_start_end(gauge_data.he)
    logger.info(f"{len(events_list)} hydrological events detected.")
    
    # remove hydro events with too much missing data
    events_list,removed_events = filter_hydro_events(events_list,gauge_data=gauge_data)
    logger.info(f"{len(removed_events)} hydrological events removed due too much missing data.")
    if keep_removed_events:
        removed_events.to_csv(output_file.parent.joinpath("events_with_too_much_missing_streamflow.csv"),index=False)
    logger.info(f"{len(events_list)} events left.")
    events_list.to_csv(temp_output_file)
    logger.info(f"Hydrological events saved to file: {temp_output_file}")
    
    # filter out events without sediment data
    events_list,removed_events = filter_events_without_sediment_data(events_list,gauge_data=gauge_data)
    logger.info(f"{len(removed_events)}, events were removed due to no sediment data.")
    if keep_removed_events:
        removed_events.to_csv(output_file.parent.joinpath("events_without_sediment_data.csv"),index=False)
    logger.info(f"{len(events_list)} events left.")
    
    # filter out events with missing sediment data, unless they are above SSC threshold
    events_list,removed_events = filter_events_with_missing_sediment_data(events_list,gauge_data=gauge_data,SSC_threshold=SSC_threshold)
    logger.info(f"{len(removed_events)} events were removed due to missing sediment data.")
    if keep_removed_events:
        # save the removed events to table
        removed_events.to_csv(output_file.parent.joinpath("events_missing_sediment_data.csv"),index=False)
    logger.info(f"{len(events_list)} events left.")
    
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

