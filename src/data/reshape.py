from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np


def _load_gsod(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = Path("../data/raw/data_candidate_stations_50km_10yr.csv")
    elif isinstance(path, str):
        path = Path(path)
    assert path.exists()
    meta = pd.DataFrame(
        {
            "stn": [
                "usaf",
                str,
            ],
            "wban": [
                "wban",
                str,
            ],
            # "year": ,  # date columns are processed via parse_dates
            # "mo": ,
            # "da": ,
            "temp": [
                "temp_f_mean",
                np.float32,
            ],
            "count_temp": [
                "temp_count",
                np.uint8,
            ],
            "dewp": [
                "dew_point_f_mean",
                np.float32,
            ],
            "count_dewp": [
                "dew_point_count",
                np.uint8,
            ],
            "slp": [
                "sea_level_pressure_mbar_mean",
                np.float32,
            ],
            "count_slp": [
                "sea_level_pressure_count",
                np.uint8,
            ],
            "stp": [
                "pressure_mbar_mean",
                np.float32,
            ],
            "count_stp": [
                "pressure_count",
                np.uint8,
            ],
            "visib": [
                "visbility_miles_mean",
                np.float32,
            ],
            "count_visib": [
                "visbility_count",
                np.uint8,
            ],
            "wdsp": [
                "wind_speed_knots_mean",
                np.float32,
            ],
            "count_wdsp": [
                "wind_speed_count",
                np.uint8,
            ],
            "max": [
                "temp_f_max",
                np.float32,
            ],
            "flag_max": [
                "temp_max_measurement_type",
                str,
            ],
            "min": [
                "temp_f_min",
                np.float32,
            ],
            "flag_min": [
                "temp_min_measurement_type",
                str,
            ],
            "prcp": [
                "precipitation_total_inches",
                np.float32,
            ],
            "flag_prcp": [
                "precipitation_measurement_type",
                str,
            ],
            "sndp": [
                "snow_depth_inches",
                np.float32,
            ],
            "rain_drizzle": [
                "had_rain",
                np.uint8,
            ],
            "snow_ice_pellets": [
                "had_snow_ice",
                np.uint8,
            ],
            "hail": [
                "had_hail",
                np.uint8,
            ],
        },
        index=["new_name", "dtype"],
    ).T
    gsod = pd.read_csv(path, parse_dates=[["year", "mo", "da"]], dtype=meta["dtype"].to_dict())
    rename_dict = meta["new_name"].to_dict()
    rename_dict["year_mo_da"] = "timestamp"
    gsod.rename(columns=rename_dict, inplace=True)
    return gsod


def _transform_gsod(gsod: pd.DataFrame) -> None:
    nominal_nan = {  # from documentation
        "temp_f_mean": 9999.9,
        "temp_f_max": 9999.9,
        "temp_f_min": 9999.9,
        "dew_point_f_mean": 9999.9,
        "sea_level_pressure_mbar_mean": 9999.9,
        "pressure_mbar_mean": 9999.9,
        "visbility_miles_mean": 999.9,
        "wind_speed_knots_mean": 999.9,
        "snow_depth_inches": 999.9,
        # note: see precipitation_measurement_type column for NaN semantics.
        # Sometimes it means 0, sometimes missing.
        "precipitation_total_inches": 99.99,
    }
    for col, sentinel_value in nominal_nan.items():
        is_nan = np.isclose(gsod.loc[:, col], sentinel_value, rtol=1e-5)
        gsod.loc[is_nan, col] = np.nan

    # fix 36 erroneous values. looks like parsing error
    gsod.loc[:, ["had_hail", "had_snow_ice"]] = gsod.loc[:, ["had_hail", "had_snow_ice"]].replace(10, 0)
    return


def get_gsod(path: Optional[Path] = None) -> pd.DataFrame:
    gsod = _load_gsod(path)
    _transform_gsod(gsod)
    return gsod
