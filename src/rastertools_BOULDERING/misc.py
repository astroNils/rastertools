from pathlib import Path

def rm_tree(pth):
    """
    Recursively remove a directory and all its contents.

    Parameters
    ----------
    pth : str or Path
        Path to the directory to be removed.

    Notes
    -----
    This function will remove all files and subdirectories within the specified path,
    then remove the directory itself.
    """
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def removekey(d, key):
    """
    Create a new dictionary with one key removed from the input dictionary.

    Parameters
    ----------
    d : dict
        Input dictionary.
    key : hashable
        Key to remove from the dictionary.

    Returns
    -------
    dict
        A new dictionary with all key-value pairs from input except the specified key.

    Examples
    --------
    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> removekey(d, 'b')
    {'a': 1, 'c': 3}
    """
    r = dict(d)
    del r[key]
    return r

def folder_structure(df, dataset_directory):
    """
    Create a standardized folder structure for image datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'dataset' column with unique dataset identifiers.
    dataset_directory : str or Path
        Base directory where the folder structure will be created.

    Notes
    -----
    Creates a two-level directory structure:
    dataset_directory/
        dataset_1/
            images/
            labels/
        dataset_2/
            images/
            labels/
        ...

    The function creates folders for each unique dataset found in df['dataset'],
    with 'images' and 'labels' subdirectories for each dataset.
    """
    dataset_directory = Path(dataset_directory)
    folders = list(df["dataset"].unique())
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)