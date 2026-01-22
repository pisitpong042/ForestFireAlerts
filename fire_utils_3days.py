"""
fire_utils_3day.py

Multi-day Fire Weather Index (FWI) computation utilities.
Refactored from fire_utils.py to remove global state and to support
multi-day time series contained in a single NetCDF file (e.g. 3 days / 72 hours).

Assumptions:
- Time dimension is hourly and divisible by 24
- Variables exist: lat, lon, T_2m, rh_2m, precip_hr, u_10m_gr, v_10m_gr
- fwi library is installed and provides fwi_ffmc, fwi_dmc, fwi_dc, fwi_isi, fwi_bui, fwi_fwi

Typical usage:

    from fire_utils_3day import compute_fwi_all_days

    days = compute_fwi_all_days("2026-01-21_00UTC_d03.nc")

    # days is a list of dicts, one per day, containing FWI fields
"""

import numpy as np
import netCDF4

from fwi import (
    bui as fwi_bui,
    dc as fwi_dc,
    dmc as fwi_dmc,
    isi as fwi_isi,
    ffmc as fwi_ffmc,
    fwi as fwi_fwi,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def detect_days(nc_file):
    """Return (nt, n_days) from a NetCDF file."""
    nc = netCDF4.Dataset(nc_file)
    nt = len(nc.variables["T_2m"])
    nc.close()
    return nt, nt // 24


# -----------------------------------------------------------------------------
# Reading one day from a multi-day NetCDF
# -----------------------------------------------------------------------------

def read_nc_multiday(nc_file, day_index, noon_hour=12):
    """
    Read one day worth of inputs from a multi-day NC file.

    Args:
        nc_file: path to NetCDF
        day_index: 0, 1, 2, ...
        noon_hour: hour index within the day used for FWI (UTC)

    Returns:
        state dict with meteorology, precipitation, and allocated output arrays
    """

    nc = netCDF4.Dataset(nc_file)

    lat = np.array(nc.variables["lat"][:])
    lon = np.array(nc.variables["lon"][:])

    t_2m = nc.variables["T_2m"]
    rh_2m = nc.variables["rh_2m"]
    precip_hr = nc.variables["precip_hr"]
    u_10m_gr = nc.variables["u_10m_gr"]
    v_10m_gr = nc.variables["v_10m_gr"]

    day_start = day_index * 24
    noon_index = day_start + noon_hour

    temp = np.array(t_2m[noon_index])
    rh   = np.array(rh_2m[noon_index])
    u    = np.array(u_10m_gr[noon_index])
    v    = np.array(v_10m_gr[noon_index])

    # wind speed km/h
    wsp = np.sqrt(u**2 + v**2) * 3.6

    # accumulate previous 24h precipitation
    precip = np.zeros_like(temp)
    start = max(0, noon_index - 24)
    for t in range(start, noon_index):
        precip += precip_hr[t]

    nc.close()

    sn, we = temp.shape

    return {
        # grid
        "lat": lat,
        "lon": lon,

        # meteorology
        "temp": temp,
        "rh": rh,
        "wsp": wsp,
        "precip": precip,

        # yesterday moisture placeholders (attached later)
        "ffmc_yda": None,
        "dmc_yda": None,
        "dc_yda": None,

        # outputs
        "ffmc": np.empty((sn, we)),
        "dmc":  np.empty((sn, we)),
        "dc":   np.empty((sn, we)),
        "isi":  np.empty((sn, we)),
        "bui":  np.empty((sn, we)),
        "fwi":  np.empty((sn, we)),
    }


# -----------------------------------------------------------------------------
# Index computations (pure functions on state dict)
# -----------------------------------------------------------------------------

def cal_ffmc(state):
    sn, we = state["ffmc"].shape
    for r in range(sn):
        for c in range(we):
            state["ffmc"][r, c] = fwi_ffmc(
                state["ffmc_yda"][r, c],
                float(state["temp"][r, c]),
                float(state["rh"][r, c]),
                float(state["wsp"][r, c]),
                float(state["precip"][r, c]),
            )


def cal_dmc(state):
    sn, we = state["dmc"].shape
    for r in range(sn):
        for c in range(we):
            state["dmc"][r, c] = fwi_dmc(
                state["dmc_yda"][r, c],
                float(state["temp"][r, c]),
                float(state["rh"][r, c]),
                float(state["precip"][r, c]),
                15, 1,
            )


def cal_dc(state):
    sn, we = state["dc"].shape
    for r in range(sn):
        for c in range(we):
            state["dc"][r, c] = fwi_dc(
                state["dc_yda"][r, c],
                float(state["temp"][r, c]),
                float(state["rh"][r, c]),
                float(state["precip"][r, c]),
                15, 1,
            )


def cal_isi(state):
    sn, we = state["isi"].shape
    for r in range(sn):
        for c in range(we):
            state["isi"][r, c] = fwi_isi(
                float(state["ffmc"][r, c]),
                float(state["wsp"][r, c]),
                True,
            )


def cal_bui(state):
    sn, we = state["bui"].shape
    for r in range(sn):
        for c in range(we):
            state["bui"][r, c] = fwi_bui(
                float(state["dmc"][r, c]),
                float(state["dc"][r, c]),
            )


def cal_fwi(state):
    sn, we = state["fwi"].shape
    for r in range(sn):
        for c in range(we):
            state["fwi"][r, c] = fwi_fwi(
                float(state["isi"][r, c]),
                float(state["bui"][r, c]),
            )


# -----------------------------------------------------------------------------
# Driver: compute FWI for all days in one file
# -----------------------------------------------------------------------------
def compute_fwi_all_days(nc_file, noon_hour=12, 
                          init_ffmc=85.0, init_dmc=6.0, init_dc=15.0):
    """
    Compute FWI sequentially for all days contained in a single NC file.

    Args:
        nc_file: path to NetCDF
        noon_hour: UTC hour index used for FWI computation
        init_ffmc, init_dmc, init_dc: initial moisture state for first day

    Returns:
        List of state dicts, one per day, containing:
            lat, lon, temp, rh, wsp, precip,
            ffmc, dmc, dc, isi, bui, fwi
    """

    nt, n_days = detect_days(nc_file)
    print(f"Detected {n_days} days ({nt} hourly steps)")

    all_days = []

    # initialise first-day moisture
    first_state = read_nc_multiday(nc_file, 0, noon_hour)
    sn, we = first_state["temp"].shape

    ffmc_yda = np.full((sn, we), init_ffmc)
    dmc_yda  = np.full((sn, we), init_dmc)
    dc_yda   = np.full((sn, we), init_dc)

    for day in range(n_days):
        print(f"\n=== Computing day {day + 1} ===")

        state = read_nc_multiday(nc_file, day, noon_hour)

        # attach yesterday moisture
        state["ffmc_yda"] = ffmc_yda
        state["dmc_yda"]  = dmc_yda
        state["dc_yda"]   = dc_yda

        # compute chain
        cal_ffmc(state)
        cal_dmc(state)
        cal_dc(state)
        cal_isi(state)
        cal_bui(state)
        cal_fwi(state)

        # ------------------------------------------------------------------
        # Explicitly keep only clean daily outputs needed downstream
        # ------------------------------------------------------------------

        daily_state = {
            # grid
            "lat":  state["lat"],
            "lon":  state["lon"],

            # meteorology (optional but useful later)
            "temp": state["temp"],
            "rh":   state["rh"],
            "wind": state["wsp"],
            "rain": state["precip"],

            # fire indices
            "ffmc": state["ffmc"],
            "dmc":  state["dmc"],
            "dc":   state["dc"],
            "isi":  state["isi"],
            "bui":  state["bui"],
            "fwi":  state["fwi"],
        }

        # store clean daily package
        all_days.append(daily_state)

        # propagate to next day
        ffmc_yda = state["ffmc"].copy()
        dmc_yda  = state["dmc"].copy()
        dc_yda   = state["dc"].copy()

    return all_days

