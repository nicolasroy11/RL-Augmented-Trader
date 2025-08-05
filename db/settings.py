import os


DB_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DB_FILE_PATH = os.path.join(DB_DIRECTORY, "ticks.duckdb")
SYMBOL = 'BTCFDUSD'