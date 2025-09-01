import importlib
from pathlib import Path
from typing import Union, List


def get_current_working_dir():
    return Path.cwd()

def get_project_root_dir(starting_path=get_current_working_dir()):
    path = starting_path
    while True:
        if (path / 'README.md').exists():
            return path
        if path.parent == path:
            break
        path = path.parent
    raise FileNotFoundError("Could not find project root dir")

def ensure_dir(dir_paths: Union[str, Path, List[Union[str, Path]]]):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path
    """
    if not isinstance(dir_paths, list):
        dir_paths = [dir_paths]

    for dir_path in dir_paths:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)