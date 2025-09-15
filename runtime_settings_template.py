debug_mode = True

dedicated_ip = '0.0.0.0'

binance_api_key =''
binance_secret_key = ''

binance_testnet_api_key =''
binance_testnet_secret_key = ''

binance_com_api_key =''
binance_com_secret_key = ''

binance_com_testnet_api_key = ''
binance_com_testnet_secret_key = ''

binance_com_futures_api_key = ''
binance_com_futures_secret_key = ''

binance_com_futures_testnet_api_key = ''
binance_com_futures_testnet_secret_key = ''

######################################################## BINANCE RUN CONFIGURATIONS ########################################################

from binance.client import Client

DATA_FREQUENCY_SECS = 5
DATA_TICKS_WINDOW_LENGTH = 150

# testnet - com
# COM_OR_US = 'com'
# IS_TESTNET = True
# BASE_ASSET = 'BTC'
# QUOTE_ASSET = 'USDT'
# IS_FUTURES = False
# SIM_ROUND_TRIP_FEE_BPS = 0

# # testnet - us
# COM_OR_US = 'us'
# IS_TESTNET = True
# BASE_ASSET = 'BTC'
# QUOTE_ASSET = 'USDC'
# IS_FUTURES = False
# SIM_ROUND_TRIP_FEE_BPS = 0

# # real trade - com
# COM_OR_US = 'com'
# IS_TESTNET = False
# BASE_ASSET = 'BTC'
# QUOTE_ASSET = 'FDUSD'
# IS_FUTURES = False
# SIM_ROUND_TRIP_FEE_BPS = 0

# # real trade - us
# COM_OR_US = 'us'
# IS_TESTNET = False
# BASE_ASSET = 'BTC'
# QUOTE_ASSET = 'USDC'
# IS_FUTURES = False
# SIM_ROUND_TRIP_FEE_BPS = 0

# futures testnet trade
COM_OR_US = 'com'
IS_TESTNET = True
BASE_ASSET = 'BTC'
QUOTE_ASSET = 'USDT'
IS_FUTURES = True
SIM_ROUND_TRIP_FEE_BPS = 4

if not IS_FUTURES:
    read_only_client = Client(testnet=IS_TESTNET, tld=COM_OR_US)
    write_client = Client(api_key=binance_testnet_api_key, api_secret=binance_testnet_secret_key, testnet=IS_TESTNET, tld=COM_OR_US)
else:
    read_only_client = Client(testnet=IS_TESTNET, tld=COM_OR_US)
    write_client = Client(api_key=binance_com_futures_testnet_api_key, api_secret=binance_com_futures_testnet_secret_key, testnet=IS_TESTNET, tld=COM_OR_US)

############################################################################################################################################

# PGSQL settings
PGSQL_DATABASE = ""
PGSQL_USER = ""
PGSQL_PASSWORD = ""
PGSQL_HOST = ""
PGSQL_PORT = 5432

# Pickle files
import os
RL_POLICY_PATH = os.path.join(os.path.dirname(__file__), 'pth_files')