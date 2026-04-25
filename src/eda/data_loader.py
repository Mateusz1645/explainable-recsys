from pathlib import Path

import pandas as pd


def load_data_tables(
    file_map: dict[str, Path],
    sep: str = "\t",
    encoding: str = "latin-1"
) -> dict[str, pd.DataFrame]:
    """Load all tables defined in ``file_map`` into memory.

    Parameters
    ----------
    file_map : dict[str, Path]
        Mapping from logical table name to file path.
    sep : str, default "\\t"
        Field separator used in input files.
    encoding : str, default "latin-1"
        Text encoding used when reading files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Loaded DataFrames keyed by table name.

    Raises
    ------
    ValueError
        If any loaded table is empty.
    """

    data: dict[str, pd.DataFrame] = {}
    for name, path in file_map.items():
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        if df.empty:
            raise ValueError(f"{name} ({path}) is empty")
        data[name] = df
    return data
