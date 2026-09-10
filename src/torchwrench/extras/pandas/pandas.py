#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from typing import Sequence, Union

import numpy as np
import pandas as pd
import pythonwrench as pw
from pandas._typing import DropKeep


def empty_dataframe(
    num_rows: int = 0,
    columns: Union[Sequence[str], pd.Series, None] = None,
) -> pd.DataFrame:
    columns = pd.Series(columns) if columns is not None else None
    df = pd.DataFrame.from_records([{}] * num_rows, columns=columns)
    return df


def drop_duplicated_col_names(
    df: pd.DataFrame,
    keep: DropKeep = "first",
) -> pd.DataFrame:
    mask_cols_to_keep = ~df.columns.duplicated(keep=keep)
    selected_columns = df.columns[mask_cols_to_keep]
    df = df[selected_columns]  # type: ignore
    return df


def concat_without_duplicated_col_names(
    dfs: Iterable[pd.DataFrame],
    *,
    verify_integrity: bool = False,
    check_values: bool = False,
) -> pd.DataFrame:
    dfs = list(dfs)

    mask_cols: list[np.ndarray] = []
    dupl_cols = set()
    prev_cols = set()

    for df_i in dfs:
        mask_cols_i = df_i.columns.isin(prev_cols) | df_i.columns.duplicated(
            keep="first"
        )
        mask_cols.append(mask_cols_i)
        dupl_cols.update([col for col in df_i.columns if col in prev_cols])
        prev_cols.update(df_i.columns)

    if check_values:
        __sanity_check(dfs, dupl_cols)

    dfs = [
        df_i.iloc[:, ~mask_cols_i]
        for df_i, mask_cols_i in zip(dfs, mask_cols, strict=True)
    ]
    df = pd.concat(
        dfs,
        axis="columns",
        ignore_index=False,
        verify_integrity=verify_integrity,
    )

    return df


def __sanity_check(
    dfs: Iterable[pd.DataFrame],
    dupl_cols: Iterable[str],
) -> None:
    for col in dupl_cols:
        values = []
        for df in dfs:
            if col not in df.columns:
                continue
            values_i: Union[pd.Series, pd.DataFrame] = df[col]

            if isinstance(values_i, pd.Series):
                values.append(values_i)
            else:
                values += [values_ij for _, values_ij in values_i.items()]

        def series_eq(x, y):
            return (x == y).all().item()

        sanity_check = pw.all_eq(values, eq_fn=series_eq)
        if sanity_check:
            continue

        concatenated = pd.concat(
            dfs, axis="columns", ignore_index=False, verify_integrity=False
        )
        diff_examples = concatenated.iloc[
            :2, np.where(concatenated.columns.isin(dupl_cols))[0]
        ]
        diff_examples_dict = diff_examples.to_dict("records")
        msg = f"Trying to remove duplicated columns '{dupl_cols}', but values are different in dataframes. (with {diff_examples_dict=})"
        raise RuntimeError(msg)
