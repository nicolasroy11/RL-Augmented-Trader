import os


DB_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DUCKDB_ARCHIVES_PATH = os.path.join(DB_DIRECTORY, "archives")
DUCKDB_FILE_PATH = os.path.join(DB_DIRECTORY, "ticks.duckdb")
SYMBOL = 'BTCFDUSD'