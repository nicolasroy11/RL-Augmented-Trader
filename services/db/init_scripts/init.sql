-- first create the rlat database via migration, then run the following

-- Set the database timezone to UTC
ALTER DATABASE rlat SET timezone TO 'UTC';

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS duckdb_transfers;

-- Add index on timestamp for performance
CREATE INDEX IF NOT EXISTS idx_btcfdusd_ticks_timestamp
ON duckdb_transfers.btcfdusd_ticks (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_btcfdusd_data_timestamp
ON public.btcfdusd_data (timestamp DESC);
