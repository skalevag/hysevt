"""
Functions to import timeseries of streamflow and suspended sediment.

author:
Amalie Skålevåg
skalevag2@uni-potsdam.de
"""

import pandas as pd
from io import StringIO
from enum import Enum
import json

# B2SHARE

def json_file_to_dict(filename: str):
    """
    Import JSON file as dictionary.
    
    Modified from: https://stackoverflow.com/a/41476738
    """
    with open(filename,"r") as f_in:
        cont = f_in.read()
        return json.loads(cont)
    
def read_B2SHARE(file):
    df = pd.read_csv(file,sep=";",index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    df.columns = ["streamflow","suspended_sediment"]
    return df

def read_reconstruction(reconstruction_file):
    df = pd.read_csv(reconstruction_file,sep="\t",index_col=0,na_values=["-999"])
    df = df[df.index != "date"]
    df.index = pd.to_datetime(df.index,format="%Y-%m-%d")
    df = df.astype(float)
    return df

# HD Tirol data

class DatasetType(Enum):
    DAILY = "daily resolution"
    SUBHOURLY = "15-min resolution"
    

def read_HD_Tirol(file, var_name: str, type: DatasetType, datetimeformat="%d.%m.%Y %H:%M:%S",returnRaw=False):
    """Reads timeseries data from Hydrographischen Dienst (HD) Tirol. 
    Extracts metadata from file header. 
    Replaces data gaps in data, and converts to pandas.DataFrame.
    
    Example:
        read_HD_Tirol("data/bedload_data.csv","bedload")
    
    Args:
        file (str, ): file
        var_name (str): _description_
        type (DatasetType): the type of dataset, i.e. the temporal resolution of the data
        datetimeformat (str, optional): _description_. Defaults to "%d.%m.%Y %H:%M:%S".
        returnRaw (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: In cases where the dataset type is not defined or supported.

    Returns:
        df (pandas.Data): _description_
        meta (str): metadata extracted from file head
    """

    with open(file, "r", encoding="latin1") as f:
        content = f.read()
    # split data into metadata and timeseries
    meta, df = content.split("Werte:\n")
    
    # return raw data if specified (for debugging)
    if returnRaw:
        return df,meta
    
    if type is DatasetType.DAILY:
        df = read_HD_Tirol_raw_daily(df)
    elif type is DatasetType.SUBHOURLY:
        df = read_HD_Tirol_raw_15min(df)
    else:
        raise TypeError("The dataset type is something I do not recognise.")
    
    # convert index to datetime format
    df.index = pd.to_datetime(df.index,format=datetimeformat)  # convert index to datetime 
    df.index.name = "time" # set new index name
    
    # rename column header
    try:
        df.columns = [var_name]
    except ValueError:
        return df
    
    return df, meta

def read_HD_Tirol_raw_daily(df: str, sep=";"):
    """Read raw data extracted from daily HD-Tirol file to dataframe.

    Args:
        df (str): dataframe as raw data
        sep (str, optional): character separating columns. Defaults to ";".

    Returns:
        pandas.Dataframe
    """
    # read raw data into dataframe object
    df = df.replace("Lücke", "NaN") # replace data gaps with NaNs
    df = pd.read_csv(StringIO(df), sep=sep, decimal=",", header=None, index_col=0,na_values=["NaN"]) # read timeseries data into dataframe
    if type(df.iloc[0, 0]) is str: # if convertion from string failed, to this "manually"
        df.iloc[:, 0] = [float(val.replace(",",".")) for val in df.iloc[:, 0].values]
    df.index = [i.strip() for i in df.index] # remove extra spaces from indeces
    return df

def read_HD_Tirol_raw_15min(df: str):
    """_summary_

    Args:
        df (str): dataframe as raw data

    Returns:
        pandas.Dataframe
    """
    df = df.replace("Lücke", "NaN") # remove data gaps
    df = df.replace(":00 ",":00 ;") # replace separator
    df = pd.read_table(StringIO(df), sep=";",decimal=",", header=None, index_col=0, na_values=["NaN"]) # read raw data into dataframe object
    df.index = [i.strip() for i in df.index] # remove extra spaces from indeces
    if type(df.iloc[0, 0]) is str: # if convertion from string failed, to this "manually"
        df.iloc[:,0] = [float(val.strip()) for val in df.iloc[:,0]]
    return df
