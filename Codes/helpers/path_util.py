# coding: utf-8

from os.path import dirname, join, normpath, exists
from os import makedirs
import time

# to get the absolute path of current project
project_root_url = normpath(join(dirname(__file__), '..'))


def from_project_root(rel_path, create=False):
    """ return system absolute path according to relative path, if path dirs not exists and create is True,
     required folders will be created

    Args:
        rel_path: relative path
        create: whether to create folds not exists

    Returns:
        str: absolute path

    """
    abs_path = normpath(join(project_root_url, rel_path))
    if create and not exists(dirname(abs_path)):
        makedirs(dirname(abs_path))
    return abs_path


def date_suffix(file_type=""):
    """ return the current date suffix，like '180723.csv'

    Args:
        file_type: file type suffix, like '.csv

    Returns:
        str: date suffix

    """
    suffix = time.strftime("%y%m%d", time.localtime())
    suffix += file_type
    return suffix


def main():
    """ for test """
    print(project_root_url)
    print(from_project_root('.gitignore'))
    print(from_project_root('data/test.py', create=False))
    print(date_suffix('.csv'))
    print(date_suffix(""))
    pass


if __name__ == '__main__':
    main()
