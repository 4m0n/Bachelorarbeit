import pandas as pd
from astropy.time import Time

from rich import print
import curves_scrapper
import calc_filter_diff
import find_variable


from rich.traceback import install
from rich.console import Console
install()
console = Console()


curves_scrapper.start()
calc_filter_diff.start()