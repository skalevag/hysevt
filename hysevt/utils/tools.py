"""
Different utility tools that didn't fit in other modules.

author:
Amalie Skålevåg
skalevag2@uni-potsdam.de
"""
import pandas as pd
import functools
import logging
from typing import Any,Callable
import time
import json



def log(logger: logging.Logger):
    def decorator(func: Callable[...,Any]) -> Callable[...,Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            logger.info(f"Calling {func.__name__}")
            logger.debug(f"Calling {func.__name__} with args {signature}")
            try:
                start = time.perf_counter()
                value = func(*args, **kwargs)
                stop = time.perf_counter()
                logger.info(f"{func.__name__} runtime: {stop-start} sec")
                return value
            except Exception as e:
                logger.exception(f"Exception raised in {func.__name__}: {str(e)}")
                raise e
        return wrapper
    return decorator


def get_freq_in_min(index: pd.DatetimeIndex) -> float:
    """Returns the temporal resolution (frequency) of a datetime index in minutes.

    Args:
        index (pd.DatetimeIndex): index containing datetime objects

    Returns:
        float: temporal resolution (frequency) in minutes
    """
    freq = pd.infer_freq(index)
    freq_in_min = pd.Timedelta(freq).total_seconds()/60
    return freq_in_min


def get_freq_in_sec(index: pd.DatetimeIndex) -> float:
    """Returns the temporal resolution (frequency) of a datetime index in seconds.

    Args:
        index (pd.DatetimeIndex): index containing datetime objects

    Returns:
        float: temporal resolution (frequency) in seconds
    """
    freq = pd.infer_freq(index)
    freq_in_sec = pd.Timedelta(freq).total_seconds()
    return freq_in_sec


def save_dict_to_json(data,file):
    with open(file, 'w') as fp:
        json.dump(data, fp, indent=4)


def load_dict_from_json(file):
    with open(file, 'r') as fp:
        data = json.load(fp)
    return data

def save_list_to_txt(someList,filepath):
    with open(filepath,"w") as f:
        f.write("\n".join(someList))
        
def load_list_from_txt(filepath):
    with open(filepath,"r") as f:
        someList = f.read().splitlines()
    return someList
