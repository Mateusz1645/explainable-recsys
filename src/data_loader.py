import pandas as pd

def load_and_validate_data(filepath: str, sep: str = '\t', columns=None, encoding='utf-8') -> pd.DataFrame:
    """
    Load a CSV/TSV file into a pandas DataFrame and perform basic validation.

    Parameters
    ----------
    filepath : str
        Path to the file to load.
    sep : str, default '\t'
        Column separator in the file.
    columns : list, optional
        List of column names to use. If None, header is inferred from the file.
    encoding : str, default 'utf-8'
        File encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data after basic validation.

    Raises
    ------
    ValueError
        If the loaded dataset is empty.
    """
    # Load the data
    df = pd.read_csv(filepath, sep=sep, names=columns, encoding=encoding, engine='python')
    
    # Check if dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty")
    
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("Missing values detected:")
        print(missing)
    else:
        print("No missing values")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    
    return df