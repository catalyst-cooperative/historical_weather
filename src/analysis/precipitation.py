from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

idx = pd.IndexSlice


def _load_erroneous_precip_points(path=None) -> pd.MultiIndex:
    """Load ids of manually determined spikes. See notebook 07 for distribution analysis."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data/processed/erroneous_precip_points.csv"
    data = pd.read_csv(path, parse_dates=["timestamp"], dtype={"usaf": str, "wban": str})
    out = pd.MultiIndex.from_frame(data)
    return out


def _load_erroneous_precip_years(path: Optional[Path] = None) -> List[Tuple[str, str, slice]]:
    """Load ids of manually determined near-zero precipitation years. See notebook 07 for analysis."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data/processed/erroneous_precip_years.csv"
    data = pd.read_csv(path, dtype=str)
    out = [(row.usaf, row.wban, slice(row.year, row.year, None)) for row in data.itertuples()]
    return out


def _find_implausible_annual_totals(df: pd.DataFrame) -> List[Tuple[str, str, slice]]:
    """Automatic detection of near-zero precipitation years. See notebook 07 for analysis."""
    annual_precip = df.groupby(
        [pd.Grouper(level="usaf"), pd.Grouper(level="wban"), pd.Grouper(level="timestamp", freq="AS")]
    )["precipitation_total_inches"].agg(["sum", "count"])
    thresh = 365 * 0.9
    annual_precip.loc[:, "sum"].where(annual_precip.loc[:, "count"].ge(thresh), inplace=True)
    # remove erroneous near-zero annual totals
    is_too_low = annual_precip.loc[:, "sum"].lt(1.0)
    to_drop = annual_precip.loc[is_too_low, :].index.to_frame()
    to_drop["timestamp"] = to_drop.loc[:, "timestamp"].dt.year.astype(str)
    slices = [(row.usaf, row.wban, slice(row.timestamp, row.timestamp, None)) for row in to_drop.itertuples()]
    return slices


def set_manual_exclusions_to_nan(
    df: pd.DataFrame,
    exclusion_idx: Union[pd.MultiIndex, List[Tuple[str, str, slice]]],
    column="precipitation_total_inches",
) -> None:
    """Apply manual data removals in place.

    Args:
        df (pd.DataFrame): GSOD subset
        exclusion_idx (Union[pd.MultiIndex, List[Tuple[str, str, slice]]]): indices of data to remove
        column (str, optional): which column to apply removals to. Defaults to "precipitation_total_inches".
    """
    if isinstance(exclusion_idx, pd.MultiIndex):
        df.loc[exclusion_idx, column] = np.nan
    elif isinstance(exclusion_idx, list):
        for slicer in exclusion_idx:
            df.loc[slicer, column] = np.nan
    return


def _test_1973_garbage_data(df: pd.DataFrame) -> pd.Series:
    """Test for presence of erroneous behavior in 1973. See notebook 07 for analysis."""
    if tuple(df.index.names) != ("usaf", "wban", "timestamp"):
        raise ValueError("Expect index of (usaf, wban, timestamp)")
    grp = df.loc[:, "precipitation_total_inches"].groupby(level=["usaf", "wban"])
    zscores = grp.apply(_precip_zscore)
    return zscores


def _precip_zscore(grp_series: pd.Series):
    """Test for presence of erroneous behavior; designed for 1973 problems."""
    # compare the first 5 months of each year
    seventy_three = grp_series.loc[idx[:, :, "1973-01-01":"1973-05-31"]]
    if len(seventy_three) < 100:  # bad coverage
        return np.nan
    the_rest = grp_series.loc[idx[:, :, "1974":]]
    the_rest = the_rest.loc[the_rest.index.get_level_values("timestamp").month <= 5]  # one-indexed
    annual_precip = the_rest.reset_index(["usaf", "wban"], drop=True).resample("AS").sum()
    annual_std_dev = annual_precip.std()
    if annual_std_dev == 0:
        return np.nan
    out = (seventy_three.sum() - annual_precip.mean()) / annual_std_dev
    return out


def remove_garbage_data_1973(df: pd.DataFrame, zscore_thresh=5):
    """Test for and remove erroneous data in the first half of 1973 (a common pattern). See notebook 07."""
    # see notebooks/07-tb-precipitation_data_cleaning.ipynb for threshold justification
    zscores = _test_1973_garbage_data(df)
    bad_stations = zscores.loc[zscores > zscore_thresh].index
    to_keep = pd.Series(np.full(df.shape[0], fill_value=True, dtype=np.bool8), index=df.index)
    for usaf, wban in bad_stations:
        to_keep.loc[idx[usaf, wban, "1973-01-01":"1973-05-31"]] = False
    out = df.loc[to_keep, :].copy()
    return out


def clean_precip_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all fixes to known precipitation data problems.

    * bad 1973 data
    * long gaps erroneously represented as zeros
    * giant erroneous spikes
    """
    if tuple(df.index.names) != ("usaf", "wban", "timestamp"):
        raise ValueError("Expect index of (usaf, wban, timestamp)")
    for exclusion_idx in (
        _load_erroneous_precip_points(),
        _load_erroneous_precip_years(),
        _find_implausible_annual_totals(df),
    ):
        set_manual_exclusions_to_nan(df, exclusion_idx=exclusion_idx, column="precipitation_total_inches")
    out = remove_garbage_data_1973(df)
    return out
