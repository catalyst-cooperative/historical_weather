from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances


def _load_gsod(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data/raw/data_candidate_stations_50km_10yr.csv"
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


def _extract_station_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = Path("../data/raw/all_gsod_stations_in_wieb_territory.csv")
    elif isinstance(path, str):
        path = Path(path)
    assert path.exists()
    dtype_dict = {
        "usaf": str,
        "wban": str,
    }
    # force ID columns to str so they don't lose zero-padding
    stations = pd.read_csv(path, parse_dates=["begin", "end"], dtype=dtype_dict)
    rename_dict = {
        "lat": "latitude",
        "lon": "longitude",
        "elev": "elevation_ft",
        "begin": "nominal_begin_date",
        "end": "nominal_end_date",
    }
    stations.rename(columns=rename_dict, inplace=True)
    return stations


def _transform_station_metadata(stations: pd.DataFrame, cities: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if cities is None:
        cities = get_cities()
    distance_matrix = _calculate_distance_matrix(stations, cities)
    nearest = _nearest_city_from_matrix(distance_matrix)
    out = pd.concat([stations, nearest], axis=1, copy=False)
    out.loc[:, "name"] = out.loc[:, "name"].str.strip()
    return out


def get_cities(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = Path("../data/processed/priority_cities.csv")
    elif isinstance(path, str):
        path = Path(path)
    assert path.exists()
    cities = pd.read_csv(path)
    return cities


def _calculate_distance_matrix(stations: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    # haversine distance gives angular distance between two points on a sphere.
    # Convert to arc length by multiplying by earth's radius
    earth_radius_km = 6371  # wikipedia

    def degrees_to_radians(df):
        return df * (np.pi / 180.0)

    station_coords = degrees_to_radians(stations.loc[:, ["latitude", "longitude"]])
    city_coords = degrees_to_radians(cities.loc[:, ["latitude", "longitude"]])
    distance_matrix = haversine_distances(station_coords, city_coords) * earth_radius_km
    distance_cols = pd.Index(cities["city"])
    distance_matrix = pd.DataFrame(distance_matrix, columns=distance_cols, index=stations.index, dtype=np.float32)
    return distance_matrix


def _nearest_city_from_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    minima = distance_matrix.min(axis=1).astype(np.float32)
    indicators = distance_matrix.eq(minima, axis=0).stack()
    # indexing by itself filters for True. Then drop the booleans.
    nearest = indicators.loc[indicators].reset_index(level="city").drop(columns=0)
    nearest.rename(columns={"city": "nearest_city"}, inplace=True)
    nearest.loc[:, "nearest_city"] = nearest.loc[:, "nearest_city"].astype(pd.CategoricalDtype())
    nearest["distance_km"] = minima
    return nearest


def get_station_metadata(station_path: Optional[Path] = None, city_path: Optional[Path] = None) -> pd.DataFrame:
    stations = _extract_station_metadata(station_path)
    cities = get_cities(city_path)
    stations = _transform_station_metadata(stations, cities)
    return stations
