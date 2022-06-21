# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

import pandas as pd

import src.data.reshape as rs
from src.analysis.precipitation import clean_precip_data


idx = pd.IndexSlice


def get_station_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data/interim/stations_for_scoping_analysis.csv"
    station_meta = pd.read_csv(path, dtype={"usaf": str, "wban": str}, index_col=["usaf", "wban"])
    return station_meta


def remove_data_pre_1973_if_gap(subset: pd.DataFrame) -> pd.DataFrame:
    annual_counts = (
        subset.loc[idx[:, :, "1970":"1977"]]
        .groupby([pd.Grouper(level="usaf"), pd.Grouper(level="wban"), pd.Grouper(level="timestamp", freq="AS")])[
            "temp_f_mean"
        ]
        .count()
    )
    count_of_counts = annual_counts.groupby([pd.Grouper(level="usaf"), pd.Grouper(level="wban")]).count()
    stations_with_gap = count_of_counts[count_of_counts < 8].index.sort_values()

    station_has_gap = subset.index.droplevel("timestamp").isin(stations_with_gap)
    is_1973_or_later = subset.index.get_level_values("timestamp") >= pd.Timestamp("1973-01-01")
    mask = ~station_has_gap | (station_has_gap & is_1973_or_later)
    return subset.loc[mask, :]


def subset_stations_again(subset: pd.DataFrame) -> pd.DataFrame:
    to_drop = {
        ("999999", "24033"),  # billings muni
        ("999999", "24131"),  # boise air terminal
        ("999999", "24221"),  # eugene mahlon
        ("999999", "23169"),  # las vegas mccarran
        ("999999", "23152"),  # hollywood burbank
        ("999999", "23174"),  # los angeles muni
        ("999999", "23183"),  # phoenix sky harbor intl ap
        ("726985", "99999"),  # portland troutdale
        ("726985", "24242"),  # portland troutdale
        ("725846", "93201"),  # truckee-tahoe
        ("725846", "99999"),  # truckee-tahoe
        ("999999", "23188"),  # SAN DIEGO LINDBERGH FIELD
        ("999999", "93107"),  # SAN DIEGO MIRAMAR NAS
        ("722930", "93107"),  # SAN DIEGO/MIRAMAR N
        ("722931", "93107"),  # MARINE CORPS AIR STATION
        ("999999", "23230"),  # OAKLAND METROPOLITAN
        ("999999", "23234"),  # SAN FRANCISCO INTL AP
        ("999999", "23239"),  # NAVAL AIR STATION
        ("745060", "23239"),  # ALAMEDA(USN)
        ("994033", "99999"),  # ALAMEDA
        ("727937", "24222"),  # SNOHOMISH CO (PAINE FD) AP
        ("727937", "99999"),  # SNOHOMISH CO
        ("999999", "23160"),  # TUCSON INTERNATIONAL AP
    }
    mask_to_keep = ~subset.index.droplevel("timestamp").isin(to_drop)
    return subset.loc[mask_to_keep, :]


def get_subset():
    gsod = rs.get_gsod()
    station_meta = get_station_metadata()
    subset_cols = [
        "timestamp",
        "temp_f_mean",
        "temp_count",
        "temp_f_max",
        "temp_f_min",
        "precipitation_total_inches",
        "temp_max_measurement_type",
        "temp_min_measurement_type",
        "precipitation_measurement_type",
    ]
    subset = (
        gsod.set_index(["usaf", "wban"])
        .loc[station_meta.index, subset_cols]
        .set_index("timestamp", append=True)
        .sort_index()
    )
    return subset


def make_dataset() -> pd.DataFrame:
    subset = get_subset()
    subset = clean_precip_data(subset)
    subset = subset_stations_again(subset)
    subset = remove_data_pre_1973_if_gap(subset)
    return subset


def main(out_path: Optional[Path] = None) -> None:
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "data/processed/historical_weather_data.csv"
    subset = make_dataset()
    subset.to_csv(out_path, index=True)


if __name__ == "__main__":
    main()
