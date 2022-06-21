from typing import Callable, Dict, List, Sequence

import pandas as pd

idx = pd.IndexSlice


def _generate_pairs_from_groups(splice_groups: pd.DataFrame) -> List[pd.DataFrame]:
    """Most splice groups are pairs, but a handful have 3 stations. Convert 3+ station groups to consecutive pairs.

    splice_groups comes from data/interim/splice_pairs.csv
    """
    pairs = []
    for _, group_df in splice_groups.set_index(["usaf", "wban"]).groupby("pair_id"):
        if len(group_df) == 2:
            pairs.append(group_df)
            continue
        for i in range(len(group_df) - 1):
            pairs.append(group_df.iloc[i : i + 2, :])
    return pairs


def _subset_daily_splice_groups(daily_df: pd.DataFrame, splice_group_df: pd.DataFrame) -> List[pd.DataFrame]:
    """Use ['usaf', 'wban'] ids from a splice_pairs.groupby('pair_id') object to subset GSOD data."""
    if daily_df.index.names != pd.core.indexes.frozen.FrozenList(["usaf", "wban", "timestamp"]):
        raise ValueError("First set index of input daily data to ['usaf', 'wban', 'timestamp']")
    if splice_group_df.index.names != pd.core.indexes.frozen.FrozenList(["usaf", "wban"]):
        raise ValueError("First set index of input splice pairs to ['usaf', 'wban']")
    indicies = [(*item, slice(None)) for item in splice_group_df.index]
    dfs = [daily_df.loc[index, :] for index in indicies]
    return dfs


def test_suite(splice_df_1: pd.DataFrame, splice_df_2: pd.DataFrame, list_of_test_funcs):
    outputs = [func(splice_df_1, splice_df_2) for func in list_of_test_funcs]
    return pd.concat(outputs)


def _window_test(
    splice_df_1: pd.DataFrame, splice_df_2: pd.DataFrame, window_years, list_of_test_funcs
) -> pd.DataFrame:
    sorted_dfs = _sort_dfs_by_max_date([splice_df_1, splice_df_2])
    early_df = sorted_dfs["df_0"]
    late_df = sorted_dfs["df_1"]
    splice_date = sorted_dfs["max_date_0"]
    next_date = _get_index_of_nearby_date(late_df, splice_date)
    late_df = late_df.iloc[next_date:, :]  # remove any earlier data
    # take a window on either side of the splice point:
    # the last part of the first df and the first part of the last df
    left_side = (
        _window(early_df, which_end="last", size_years=window_years)
        .resample("365d", origin="end", level="timestamp")
        .mean()
    )
    right_side = (
        _window(late_df, which_end="first", size_years=window_years)
        .resample("365d", origin="end", level="timestamp")
        .mean()
    )
    outputs = [func(left_side, right_side) for func in list_of_test_funcs]
    return outputs


def _window(df: pd.DataFrame, which_end: str, size_years: int) -> pd.DataFrame:
    delta = pd.Timedelta(f"{size_years*365 - 1}d")
    date_index = df.index.get_level_values(_get_multiindex_datetimeindex_name(df))
    if which_end == "first":
        start = date_index.min()
        end = start + delta
    elif which_end == "last":
        end = date_index.max()
        start = end - delta
    return df.loc[idx[:, :, start:end], :]


def _get_index_of_nearby_date(df: pd.DataFrame, date_string: str, method: str = "bfill") -> int:
    # methods: "bfill" = closest future date, "ffill" = closest prior date, "nearest" = nearest
    if isinstance(df.index, pd.MultiIndex):
        df_index = df.index.get_level_values(_get_multiindex_datetimeindex_name(df))
    else:
        df_index = df.index
    datetime = pd.to_datetime(date_string)
    index_number = df_index.get_indexer([datetime], method=method)[0]
    return index_number


def _get_multiindex_datetimeindex_name(df: pd.DataFrame) -> str:
    idx_of_datetime = [i for i, val in enumerate(df.index.dtypes.values) if pd.api.types.is_datetime64_dtype(val)][0]
    name = df.index.names[idx_of_datetime]
    return name


def _sort_dfs_by_max_date(dfs: Sequence[pd.DataFrame]) -> Dict[pd.Timestamp, pd.DataFrame]:
    max_dates = [df.index.get_level_values(_get_multiindex_datetimeindex_name(df)).max() for df in dfs]
    end_date_dict = dict(zip(max_dates, dfs))
    lowest_first = sorted(end_date_dict.keys())
    out = {}
    for i, key in enumerate(lowest_first):
        out[f"max_date_{i}"] = key
        out[f"df_{i}"] = end_date_dict[key]
    return out


def _bootstrap_ci(
    period_1: pd.DataFrame, period_2: pd.DataFrame, n_samples: int, quantiles: Sequence[float] = (0.05, 0.95)
):
    def apply_func(df: pd.DataFrame):
        return df.agg(["mean", "std"]).unstack()

    bootstrapped_means_1 = _bootstrap_stat(period_1, n_samples=n_samples, stat_func=apply_func)
    bootstrapped_means_2 = _bootstrap_stat(period_2, n_samples=n_samples, stat_func=apply_func)
    diff = bootstrapped_means_2.sub(bootstrapped_means_1)
    # Confidence Interval
    ci = diff.quantile(quantiles).unstack()
    return ci, diff


def _bootstrap_stat(
    source_df: pd.DataFrame, n_samples: int, stat_func: Callable[[pd.DataFrame], pd.Series]
) -> pd.Series:
    def sample(df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1, replace=True, ignore_index=True)

    stats = [stat_func(sample(source_df)) for _ in range(n_samples)]
    return pd.concat(stats, axis=1).T
