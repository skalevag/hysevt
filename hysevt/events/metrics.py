"""
Extract hydro-sedimentary events and calculate metrics for a given a list of events, and streamflow and suspended sediment observations.

Given a table of event start and end times, and a timeseries of streamflow and suspended sediment concentration, 
each event is extracted and characterised and the calculacted metrics/indices are stored in a table (pandas.DataFrame), 
where each row represents one event.

author:
Amalie Skålevåg
skalevag2@uni-potsdam.de
"""

# modules
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.signal
import subprocess

from hysevt.utils import conversion, tools
import logging
from hysevt.utils.tools import log

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


####### GET EVENTS AND EVENTS DATA #######

# events
def get_events(file) -> pd.DataFrame:
    """Get list of events from file.

    Args:
        file (str, pathlib.Path): filepath

    Returns:
        pd.DataFrame: list of events with start and end dates as datetime
    """
    # get event start and end timesteps from file
    events_list = pd.read_csv(file)
    # convert all columns to datetime
    events_list.start = pd.to_datetime(events_list.start)
    events_list.end = pd.to_datetime(events_list.end)
    return events_list


def get_gauge_data(file, column_streamflow="streamflow", column_suspended_sediment="suspended_sediment") -> pd.DataFrame:
    """Gets gauging station streamflow and suspended sediment concentration data from file.
    Returns a dataframe with datetime index that only contains streamflow and suspended sediment concentrations.

    Args:
        file (str, pathlib.Path): filepath
        column_streamflow (str, optional): name of column containing streamflow data. Defaults to "streamflow".
        column_suspended_sediment (str, optional): name of column containing suspended sediment concentration data. Defaults to "suspended_sediment".

    Returns:
        pd.DataFrame: timeseries of gauging station data, with columns:
            "streamflow": streamflow/discharge
            "suspended_sediment": suspended sediment concentrations
    """
    # get timeseries of streamflow and suspended sediment from gauging station
    data = pd.read_csv(file, index_col=0)  # load data
    data.index = pd.to_datetime(data.index)  # set index to datetime
    # get only the streamflow and suspended sediment data
    data = data[[column_streamflow, column_suspended_sediment]]
    # set the column names so that it always consistent
    data.columns = ["streamflow", "suspended_sediment"]
    return data


def get_event_data(start: pd.Timestamp, end: pd.Timestamp, data: pd.DataFrame) -> pd.DataFrame:
    """Extract the event timeseries from a longer timeseries.

    Args:
        start (pd.Timestamp): start of the event
        end (pd.Timestamp): end of the event
        data (pd.DataFrame): gauging station data from which to extract the event, must have a DatetimeIndex

    Returns:
        pd.DataFrame: timeseries of event
    """
    # extract the event timeseries
    return data[start:end]


def get_event_series(events_list: pd.DataFrame, data: pd.DataFrame) -> list:
    """_summary_

    Args:
        events_list (pd.DataFrame): list of events with start and end dates as datetime, must contain the following
            "start": column with pd.Timestamp objects
            "end": column with pd.Timestamp objects
        data (pd.DataFrame): gauging station data from which to extract the event, must have a DatetimeIndex

    Returns:
        list: of event timeseries (pandas.Dataframe),  with DatetimeIndex
    """
    event_series = []
    for i, event in events_list.iterrows():
        event_series.append(get_event_data(event.start, event.end, data))
    return event_series


####### CHARACTERISE EVENTS #######

# duration and seasonal timing of event
def event_duration(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Duration of an event in hours.

    Args:
        start (pd.Timestamp): start of the event
        end (pd.Timestamp): end of the event

    Returns:
        float: duration of event in hours
    """
    # calculates event duration in hours
    duration = (end - start).total_seconds() / 3600  # in hours
    return duration

def seasonal_timing(start_time: pd.Timestamp,end_time: pd.Timestamp) -> float:
    """Calculate metric of seasonal timing in day of year of event. Average of start and end day.

    Args:
        start_time (pd.Timestamp): start of the event
        end_time (pd.Timestamp): end of the event

    Returns:
        float: average day of year (seasonal timing) of event
    """
    return np.mean([start_time.dayofyear,end_time.dayofyear])

def seasonality_winter_summer(doy: float) -> float:
    """Calculate metric of seasonal timing, i.e. cosine of day of year. 
    Higher values express "mid-winterness" and lower values "mid-summerness".
    DOY 1, i.e. 1 January, corresponds to 1.
    DOY 183, i.e. 1 July, corresponds to -1.

    Args:
        doy (float)): seasonal timing of event, average day of year

    Returns:
        float
    """
    return np.cos(np.deg2rad((doy/366)*360))

def seasonality_spring_autumn(doy: float) -> float:
    """Calculate metric of seasonal timing, i.e. sine of day of year. 
    Higher values express "mid-springness" and lower values "mid-autumness".
    DOY 92, i.e. 1 Aprul, corresponds to 1.
    DOY 275, i.e. 1 October, corresponds to -1.

    Args:
        doy (float)): seasonal timing of event, average day of year

    Returns:
        float
    """
    return np.sin(np.deg2rad((doy/366)*360))

# suspended sediment
def total_suspended_sediment_yield(
    event_series: pd.DataFrame,
    freq_in_min: float,
    column_streamflow="streamflow",
    column_suspended_sediment="suspended_sediment",
) -> float:
    """Total suspended sediment yield of an event (in tonnes).

    Args:
        event_series (pd.DataFrame): streamflow and suspended sediment concentration timeseries of event, must have a DatetimeIndex
        freq_in_min (float or int): temporal resolution (frequency) in minutes
        column_streamflow (str, optional): name of column containing streamflow data given in m3 per second. Defaults to "streamflow".
        column_suspended_sediment (str, optional): name of column containing suspended sediment concentration data given in mg per liter. Defaults to "suspended_sediment".

    Returns:
        float: total suspended sediment yield of event in tonnes
    """
    # calculate event suspended sediment yield
    SSYtot = conversion.sediment_yield(
        event_series[column_suspended_sediment],
        event_series[column_streamflow],
        freq_in_min=freq_in_min,
    ).sum()
    return SSYtot

def proportion_of_annual_suspended_sediment_yield(
    event_series: pd.DataFrame,
    annualSSY: float,
    freq_in_min: float,
    column_streamflow="streamflow",
    column_suspended_sediment="suspended_sediment",
) -> float:
    """Total suspended sediment yield of an event as proportion of annual sediment yield.

    Args:
        event_series (pd.DataFrame): streamflow and suspended sediment concentration timeseries of event, must have a DatetimeIndex
        annualSSY (float): annual suspended sediment yield in tonnes
        freq_in_min (float): temporal resolution (frequency) in minutes
        column_streamflow (str, optional): name of column containing streamflow data given in m3 per second. Defaults to "streamflow".
        column_suspended_sediment (str, optional): name of column containing suspended sediment concentration data given in mg per liter. Defaults to "suspended_sediment".

    Returns:
        float: event suspended sediment yield as a proportion of annual suspended sediment yield
    """
    SSYtot = total_suspended_sediment_yield(
        event_series, 
        freq_in_min, 
        column_streamflow=column_streamflow,
        column_suspended_sediment=column_suspended_sediment
    )
    pSSY = SSYtot / annualSSY
    return pSSY

def peak_suspended_sediment_concentration(ssc_series: pd.Series) -> float:
    """Peak (i.e. maximum) suspended sediment concentration of an event.

    Args:
        ssc_series (pd.Series): suspended sediment concentration timeseries given in mg per liter

    Returns:
        float: peak(maximum) suspended sediment concentration
    """
    return ssc_series.max()

def mean_suspended_sediment_concentration(ssc_series: pd.Series) -> float:
    """Mean suspended sediment concentration of an event.

    Args:
        ssc_series (pd.Series): suspended sediment concentration timeseries given in mg per liter

    Returns:
        float: mean suspended sediment concentration
    """
    return ssc_series.mean()

def get_suspended_sediment_peaks(
    ssc_series: pd.Series,
    distance=4,
    prominence=500,
    height=None,
    threshold=None,
) -> pd.Series:
    """Identify suspended sediment peaks using scipy.signal.find_peaks

    Args:
        ssc_series (pd.Series): suspended sediment concentration timeseries given in mg per liter
        distance (int, optional): see scipy.signal.find_peaks for details. Defaults to 4.
        prominence (int, optional): see scipy.signal.find_peaks for details. Defaults to 500.
        threshold (int, optional): see scipy.signal.find_peaks for details. Defaults to None.
        height (_type_, optional): see scipy.signal.find_peaks for details. Defaults to None.

    Returns:
        pd.Series: values and timing(index) of peaks
    """
    # get the SSC peaks
    peaks, _ = scipy.signal.find_peaks(
        ssc_series,
        threshold=threshold,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    # return timesteps with peaks
    if len(peaks) == 0:
        return ssc_series[ssc_series==ssc_series.max()]
    else:
        return ssc_series.iloc[peaks]

def number_suspended_sediment_peaks(
    ssc_series: pd.Series,
    distance=4,
    prominence=500,
    height=None,
    threshold=None,
) -> int:
    """Number of suspended sediment peaks found using scipy.signal.find_peaks

    Args:
        ssc_series (pd.Series): suspended sediment concentration timeseries given in mg per liter
        distance (int, optional): see scipy.signal.find_peaks for details. Defaults to 4.
        prominence (int, optional): see scipy.signal.find_peaks for details. Defaults to 500.
        threshold (int, optional): see scipy.signal.find_peaks for details. Defaults to None.
        height (_type_, optional): see scipy.signal.find_peaks for details. Defaults to None.

    Returns:
        int: number of peaks
    """
    # get the SSC peaks
    peaks = get_suspended_sediment_peaks(
        ssc_series,
        threshold=threshold,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    # return number of peaks
    return len(peaks)


# streamflow/discharge
def total_streamflow_volume(q_series: pd.Series, freq_in_sec: int) -> float:
    """Total streamflow volume of an event (in cubicmeters)

    Args:
        q_series (pd.Series): streamflow timeseries given in m3 per second
        freq_in_sec (int or float): temporal resolution (frequency) in seconds

    Returns:
        float: streamflow volume in m3
    """
    return (q_series * freq_in_sec).sum()

def proportion_of_annual_streamflow_volume(
    q_series: pd.Series, annualQvol, freq_in_sec
) -> float:
    """Total streamflow volume of an event as proportion of annual streamflow volume.

    Args:
        q_series (pd.Series): streamflow timeseries given in m3 per second
        annualQvol (_type_): annual streamflow volume in m3
        freq_in_sec (_type_): temporal resolution (frequency) in seconds

    Returns:
        float: event streamflow volume as a proportion of annual streamflow volume
    """
    Qvol = total_streamflow_volume(
        q_series, freq_in_sec=freq_in_sec
    )
    pQvol = Qvol / annualQvol 
    return pQvol

def peak_streamflow(q_series: pd.Series) -> float:
    """The peak or maximum streamflow of a streamflow timeseries.

    Args:
        q_series (pd.Series): streamflow timeseries

    Returns:
        float: peak streamflow
    """
    return q_series.max()

def mean_streamflow(q_series: pd.Series) -> float:
    """The average of a streamflow timeseries.

    Args:
        q_series (pd.Series): streamflow timeseries

    Returns:
        float: mean streamflow
    """
    return q_series.mean()

def get_streamflow_peaks(
    q_series: pd.Series,
    distance=4,
    prominence=2,
    threshold=None,
    height=None,
) -> pd.Series:
    # get the streamflow peaks
    peaks, _ = scipy.signal.find_peaks(
        q_series,
        threshold=threshold,
        height=height,
        distance=distance,
        prominence=prominence
    )
    # return timesteps with peaks
    if len(peaks) == 0:
        return q_series[q_series==q_series.max()]
    else:
        return q_series.iloc[peaks]

def number_streamflow_peaks(
    q_series: pd.Series,
    distance=4,
    prominence=2,
    threshold=None,
    height=None,
) -> int:
    # get the streamflow peaks
    peaks = get_streamflow_peaks(
        q_series,
        threshold=threshold,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    # return number of peaks
    return len(peaks)

def peak_phase_difference(
    event_series,
    column_streamflow="streamflow",
    column_suspended_sediment="suspended_sediment",
    ):
    # from Haddadchi & Hicks 2021
    t_SSCmax = event_series[event_series[column_suspended_sediment] == event_series[column_suspended_sediment].max()][column_suspended_sediment].index[0]
    t_Qmax = event_series[event_series[column_streamflow] == event_series[column_streamflow].max()][column_streamflow].index[0]
    peak_phase_diff = ((t_Qmax - t_SSCmax).total_seconds()/3600) / event_duration(event_series.index[0], event_series.index[-1])
    return peak_phase_diff

def get_rising_falling(event_series,column_streamflow="streamflow"):
    peak_timing = event_series.index[
        event_series[column_streamflow] == peak_streamflow(event_series[column_streamflow])
    ][0]
    rising = event_series.index <= peak_timing
    falling = np.logical_not(rising)

    return event_series[rising], event_series[falling]

def falling_to_rising_SSY_ratio(event_series,freq_in_min,column_streamflow="streamflow",column_suspended_sediment="suspended_sediment"):
    # get rising and falling limb of event
    rising_series, falling_series = get_rising_falling(
        event_series,column_streamflow=column_streamflow
    )
    
    # suspended sediment yield in rising and falling limbs
    falling_to_rising_ratio = np.log(
        total_suspended_sediment_yield(
        falling_series, freq_in_min=freq_in_min, column_suspended_sediment=column_suspended_sediment
    ) / total_suspended_sediment_yield(
        rising_series, freq_in_min=freq_in_min, column_suspended_sediment=column_suspended_sediment
    ))
    
    return falling_to_rising_ratio

def falling_to_rising_volume_ratio(event_series, freq_in_sec,column_streamflow="streamflow"):
    # get rising and falling limb of event
    rising_series,falling_series = get_rising_falling(
        event_series,column_streamflow=column_streamflow
    )

    # falling to rising streamflow volume ratio
    falling_to_rising_ratio = np.log(
        total_streamflow_volume(
            falling_series[column_streamflow],
            freq_in_sec=freq_in_sec,
        )
        / total_streamflow_volume(
            rising_series[column_streamflow],
            freq_in_sec=freq_in_sec,
        )
    )

    return falling_to_rising_ratio
    
# hysteresis
@log(logger)
def call_hysteresis_index_script(path_gauge_data: Path,path_event_list: Path, path_rscript=Path(__file__).parent.joinpath("hysteresis_index.R"),save_hysteresis_plots=False):
    """Calls an R-script which calculates 2 hysteresis indeces.

    Args:
        path_gauge_data (pathlib.Path): absolute path to csv file containing gaunging station data
        path_event_list (pathlib.Path): absolute path to csv file containing event start and end timestamps
        path_rscript (pathlib.Path, optional): path to R script calculating the hysteresis indeces. Defaults to Path(__file__).parent.joinpath("hysteresis_index.R").
    
    References:
    @computer_program{Tsyplenkov2022,
        author = {Anatolii Tsyplenkov},
        doi = {10.5281/ZENODO.6992087},
        month = {8},
        title = {atsyplenkov/loadflux: Zenodo release},
        url = {https://zenodo.org/record/6992087},
        year = {2022},
    }
    """
    if not path_gauge_data.is_file():
        raise FileNotFoundError(f"{path_gauge_data}")
    elif not path_event_list.is_file():
        raise FileNotFoundError(f"{path_event_list}")
    elif not path_rscript.is_file():
        raise FileNotFoundError(f"{path_rscript}")
    
    if save_hysteresis_plots:
        pdf = "TRUE"
        logger.info(f"PDF of hysteresis plots will be saved at output location.")
    else:
        pdf = "FALSE"
        logger.info(f"PDF of hysteresis plots will not be generated.")

    # create command
    cmd = ["Rscript", f"{path_rscript.resolve()}", f"{path_gauge_data.resolve()}", f"{path_event_list.resolve()}",pdf]
    logger.info(f"Calling R-script with CMD : {' '.join(cmd)}")
    # run process
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logger.info(out.stdout)
    outfile = Path(str(path_event_list).split(".")[0]+"_hysteresis_index.csv")
    return outfile

@log(logger)
def calc_hysteresis_index(path_gauge_data: Path,path_event_list: Path, save_hysteresis_plots=False, path_rscript=Path(__file__).parent.joinpath("hysteresis_index.R")) -> pd.DataFrame:
    """Calculates hysteresis indeces (AHI,SHI) for all events in list.

    Args:
        path_gauge_data (pathlib.Path): absolute path to csv file containing gaunging station data
        path_event_list (pathlib.Path): absolute path to csv file containing event start and end timestamps

    Returns:
        pandas.DataFrame: hysteresis indeces AHI and SHIfor all events

    References:
    @computer_program{Tsyplenkov2022,
        author = {Anatolii Tsyplenkov},
        doi = {10.5281/ZENODO.6992087},
        month = {8},
        title = {atsyplenkov/loadflux: Zenodo release},
        url = {https://zenodo.org/record/6992087},
        year = {2022},
    }
    """
    outfile = call_hysteresis_index_script(path_gauge_data,path_event_list,save_hysteresis_plots=save_hysteresis_plots,path_rscript=path_rscript)
    logger.info(f"Results saved: {outfile}")
    results = pd.read_csv(outfile)
    results.start = pd.to_datetime(results.start)
    results.end = pd.to_datetime(results.end)
    return results

# magnitude to elapsed time logratio
def append_mag_time_logratio(event_metrics: pd.DataFrame, magnitude_metric: str) -> pd.DataFrame:
    """Inter-event index (IEI), i.e. logratio of last event magnitude to time since last event, will be appended to existing event metrics dataframe.

    AKA: Skålevåg-Schmidt metric

    Args:
        event_metrics (pd.DataFrame): dataframe of event metrics, each row is an event
        magnitude_metric (str): magnitude metric to use in ratio, e.g. suspended sediment yield, or total streamflow volume
        mag_annual_average (float): the annual average of magnitude metric

    Returns:
        pd.DataFrame: event metrics dataframe with the logratio appended
    """
    mag_time_ratio = [
        np.nan,
    ]  # first one is nan since there is no preceeding event
    for i in range(1, len(event_metrics)):  # start with second event in dataframe
        delta_time = event_metrics.iloc[i, :].time_since_last_event  # current event
        mag_last = event_metrics.iloc[i - 1, :][magnitude_metric] # last event
        mag_time_ratio.append(
            np.log(mag_last / delta_time) 
        )  # magnitude to time since last logratio
    assert len(mag_time_ratio) == len(event_metrics)  # check that lengths match
    event_metrics[
        f"IEI_{magnitude_metric}"
    ] = mag_time_ratio  # append to event metrics dataframe
    return event_metrics

# master function
@log(logger)
def calculate_event_metrics(
    events_list: pd.DataFrame,
    gauge_data: pd.DataFrame,
    min_peak_distance=4,
    SSC_peak_prominence=500,
    Q_peak_prominence=2,
    column_streamflow="streamflow",
    column_suspended_sediment="suspended_sediment",
    catchment_area=None
) -> pd.DataFrame:
    """Calculates metrics for all events in a given list of events.

    Args:
        events_list (pd.DataFrame): dataframe containing start and end dates for events
        data (pd.DataFrame): gauging station data of streamflow and suspended sediment
        annual_sediment_yield (pd.DataFrame): dataframe with annual sediment yields
        annual_streamflow_volume (pd.DataFrame): dataframe with annual streamflow volume
        column_streamflow (str, optional): name of streamflow column in gauging station data. Defaults to "streamflow".
        column_suspended_sediment (str, optional): name of suspended sediment column in gauging station data. Defaults to "suspended_sediment".
        minimum_event_length (int, optional): minimum amount of time steps required in the event time series. Defaults to 12.
        add_inter_event_effect_metrics (bool, optional): whether or not to add metrics related to inter-event effects, i.e. previous event. Defaults to True.

    Returns:
        pd.DataFrame: table with event metrics
    """
    # ensure that event list is datetime objects
    events_list = events_list.apply(pd.to_datetime)

    # temporal resolusion of data
    freq_in_min = tools.get_freq_in_min(gauge_data.index)
    freq_in_sec = tools.get_freq_in_sec(gauge_data.index)
    
    # metrics to be calculated from event time series
    SSY = []
    SSC_peak = []
    SSC_mean = []
    n_SSC_peaks = []
    Qtotal = []
    Q_peak = []
    Q_mean = []
    n_Q_peaks = []
    peak_phase_diff = []
    fall_to_risi_SSY_ratio = []
    fall_to_risi_volume_ratio = []

    # iterate over each event identified in the events list
    for i, event in events_list.iterrows():
        event_series = get_event_data(event.start, event.end, gauge_data)
        
        # suspended sediment magnitude
        SSY.append(
            total_suspended_sediment_yield(event_series, freq_in_min=freq_in_min)
        )
        SSC_peak.append(peak_suspended_sediment_concentration(event_series[column_suspended_sediment]))
        SSC_mean.append(mean_suspended_sediment_concentration(event_series[column_suspended_sediment]))

        # streamflow magnitude
        Qtotal.append(
            total_streamflow_volume(
                event_series[column_streamflow], freq_in_sec=freq_in_sec
            )
        )
        Q_peak.append(peak_streamflow(event_series[column_streamflow]))
        Q_mean.append(mean_streamflow(event_series[column_streamflow]))
        
        # other metrics
        n_SSC_peaks.append(number_suspended_sediment_peaks(event_series[column_suspended_sediment],distance=min_peak_distance,prominence=SSC_peak_prominence))
        n_Q_peaks.append(number_streamflow_peaks(event_series[column_streamflow],distance=min_peak_distance,prominence=Q_peak_prominence))
        peak_phase_diff.append(peak_phase_difference(event_series,column_streamflow=column_streamflow,column_suspended_sediment=column_suspended_sediment))
        fall_to_risi_SSY_ratio.append(falling_to_rising_SSY_ratio(event_series,freq_in_min=freq_in_min,column_streamflow=column_streamflow,column_suspended_sediment=column_suspended_sediment))
        fall_to_risi_volume_ratio.append(falling_to_rising_volume_ratio(event_series, freq_in_sec=freq_in_sec,column_streamflow=column_streamflow))

    
    # add results for each event to output
    event_metrics = events_list.copy()
    event_metrics["duration"] = [
        event_duration(start, end)
        for (start, end) in zip(events_list.start, events_list.end)
    ]
    event_metrics["seasonal_timing"] = [seasonal_timing(start,end) for (start, end) in zip(events_list.start, events_list.end)]
    event_metrics["seasonality_winter_summer"] = seasonality_winter_summer(event_metrics["seasonal_timing"])
    event_metrics["seasonality_spring_autumn"] = seasonality_spring_autumn(event_metrics["seasonal_timing"])
    event_metrics["year"] = events_list.start.apply(lambda x: x.year)
    event_metrics["SSY"] = SSY
    if catchment_area is not None:
        event_metrics["sSSY"] = event_metrics["SSY"]/catchment_area
    event_metrics["SSYn"] = event_metrics["SSY"]/event_metrics["duration"]
    event_metrics["SSC_max"] = SSC_peak
    event_metrics["SSC_mean"] = SSC_mean
    event_metrics["SSC_mean_weighted"] = (np.array(SSY) / np.array(Qtotal)) * 10**6 # from Haddadchi & Hicks 2021
    event_metrics["n_SSC_peaks"] = n_SSC_peaks
    event_metrics["Qtotal"] = Qtotal
    event_metrics["Q_max"] = Q_peak
    event_metrics["Q_mean"] = Q_mean
    event_metrics["n_Q_peaks"] = n_Q_peaks 
    event_metrics["peak_phase_diff"] = peak_phase_diff 
    event_metrics["SSY_falling_to_rising_ratio"] = fall_to_risi_SSY_ratio
    event_metrics["Qtotal_falling_to_rising_ratio"] = fall_to_risi_volume_ratio
    event_metrics["SQPR"] = np.log(event_metrics["n_SSC_peaks"]/event_metrics["n_Q_peaks"]) # ratio of suspended sediment to streamflow peaks
    event_metrics["Q_max_previous_ratio"] = np.log(event_metrics.Q_max.shift(1)/event_metrics.Q_max)
    time_since_last_event = []
    for i,(_,event) in enumerate(events_list.iterrows()):
        if i == 0:
            time_since_last_event.append(np.nan)
        else:
            time_since_last_event.append((event.start - event_metrics.iloc[i-1].end).total_seconds() / 3600)
    event_metrics["time_since_last_event"] = time_since_last_event
    append_mag_time_logratio(event_metrics,magnitude_metric="SSY")
    append_mag_time_logratio(event_metrics,magnitude_metric="Qtotal")
    append_mag_time_logratio(event_metrics,magnitude_metric="Q_max")
    append_mag_time_logratio(event_metrics,magnitude_metric="SSC_max")

    return event_metrics


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
    data = get_gauge_data(file_gauge_data)
    events_list = pd.read_csv(file_events)
    annual_sediment_yield = pd.read_csv(file_annual_sediment,index_col=0)
    annual_streamflow_volume = pd.read_csv(file_annual_streamflow,index_col=0)
    # calucalte the metrics
    event_metrics = calculate_event_metrics(events_list,data,annual_sediment_yield,annual_streamflow_volume)
    
    hysteresis_index = calc_hysteresis_index(file_gauge_data,file_events)
    event_metrics = event_metrics.join(hysteresis_index.drop(columns=["start","end"]))
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