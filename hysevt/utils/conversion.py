"""
Functions used convert between different variables.

author:
Amalie Skålevåg
skalevag2@uni-potsdam.de
"""

def sediment_conc(sediment_yield_in_t, q_in_m3_per_sec, freq_in_min):
    ssc_in_mg_per_l = sediment_yield_in_t / (
        q_in_m3_per_sec * freq_in_min * 60 * 10 ** -6
    )
    return ssc_in_mg_per_l

def sediment_yield(ssc_in_mg_per_l, q_in_m3_per_sec, freq_in_min):
    sediment_yield_in_t = (
        ssc_in_mg_per_l * q_in_m3_per_sec * freq_in_min * 60 * 10 ** -6
    )
    return sediment_yield_in_t

def water_yield(q_in_m3_per_sec, freq_in_min):
    water_yield_in_m3 = (
        q_in_m3_per_sec * freq_in_min * 60
    )
    return water_yield_in_m3