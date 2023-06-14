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
import watersedimentpulses.events.metrics
import logging
from watersedimentpulses.utils.tools import log

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
def local_minimum(path_gauge_data: Path,output_file: Path, event_station_id: str, event_version_id = "AUT", ma_window = "3H",keep_ma_gauge_data = False, filter_SSC_quantile=None) -> pd.DataFrame:
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
    logger.info(f"Output file: {output_file}")
    logger.info(f"Rolling median window: {ma_window}")
    if not filter_SSC_quantile is None:
        logger.info(f"filter_SSC_quantile = {filter_SSC_quantile}")
    if keep_ma_gauge_data:
        logger.info(f"Keeping {ma_window}-median smoothed gauge data.")
    
    # get gaugin station data
    gauge_data = watersedimentpulses.events.metrics.get_gauge_data(path_gauge_data)
    
    # apply moving median smoothing
    gauge_data_smooth = gauge_data.rolling(ma_window,center=True).median()
    
    # loc to save smooth data
    path_gauge_data_smooth = path_gauge_data.parent.joinpath(f"{path_gauge_data.name.split('.')[0]}_{ma_window}-median_smooth.csv")
    gauge_data_smooth.to_csv(path_gauge_data_smooth)
    
    
    # make temp filename
    temp_output_file = Path(str(output_file).split(".")[0]+"_all_hydro_events.csv")
    call_local_minimum_script(path_gauge_data,temp_output_file)
    
    if not keep_ma_gauge_data:
        # remove smoothed gauge data 
        path_gauge_data_smooth.unlink()
    
    # import list of hydro events
    events_list = pd.read_csv(temp_output_file)
    logger.info(f"{len(events_list)} hydrological events detected.")
    
    # filter out events without sediment data
    events_series = watersedimentpulses.events.metrics.get_event_series(events_list,gauge_data) # timeseries for each event
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
    events_series = watersedimentpulses.events.metrics.get_event_series(events_list,gauge_data)
    event_numbers = events_list.index.to_list()
    # loop through events again
    events_missing_sediment = []
    high_events_missing_sediment = []
    for i,series in zip(event_numbers,events_series):
        if series.isna().any().any():
            # high magnitude events with missing data are kep
            if series.suspended_sediment.max() > 1000:
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
    
    if not filter_SSC_quantile is None:
        SSC_threshold = gauge_data.suspended_sediment.quantile(filter_SSC_quantile)
        events_series = watersedimentpulses.events.metrics.get_event_series(events_list,gauge_data)
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

