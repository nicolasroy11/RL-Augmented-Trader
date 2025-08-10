from typing import List, Set
from endpoints import cli as endpoints_cli
from dtos import cli as dtos_cli
from enums import cli as enums_cli
import codegen_settings as settings
import os
import warnings
from glob import glob

# make sure the directory exists
if not os.path.exists(settings.PROJECT_ROOT):
    warnings.warn('The root directory you supplied does not exist')
    raise

# gather all py files in the project root recursively
search_set: Set[str] = set(glob(settings.PROJECT_ROOT + '/**/[!_]*.py', recursive=True))

# subtract excluded directories from search set
for excl in settings.EXCLUDED_DIRECTORIES:
    excluded_dir: List[str] = glob(settings.PROJECT_ROOT + '/' + excl + '/**/[!_]*.py', recursive=True)
    search_set = set(search_set - set(excluded_dir))

# add all the included directories to search set
for incl in settings.INCLUDED_DIRECTORIES:
    included_dir: List[str] = glob(incl + '/**/[!_]*.py', recursive=True)
    search_set = search_set.union(set(included_dir))
    # search_set.extend(incl)

if __name__ == "__main__":
    # generates the dtos
    dtos_cli.main(search_set)

    # generates the enums
    enums_cli.main(search_set)

    # generates the api methods
    endpoints_cli.main(search_set)