from pathlib import Path

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def folder_structure(df, dataset_directory):
    dataset_directory = Path(dataset_directory)
    folders = list(df["dataset"].unique())
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)