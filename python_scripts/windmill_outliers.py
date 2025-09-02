"""
ForVARD Project - Windmill-Compatible Outliers Detection
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics. 
Research code - Unauthorized distribution prohibited.

Windmill-compatible version of the original outliers_detection.py script.
Preserves IDENTICAL logic but adapts for Windmill execution with S3 storage.
"""

import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from numba import njit, prange
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os 
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import pyarrow.parquet as pq
import io
import pyarrow as pa

# Windmill imports
import wmill

# S3/MinIO imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import requests

# Logger configuration - CONSOLE OUTPUT for Windmill
logger = logging.getLogger("OutliersProcessor")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ============================================================
# SLACK NOTIFICATION FUNCTION
# ============================================================

def send_slack_notification(message_text):
    """
    Sends a notification message to a configured Slack channel.
    
    Reads the Slack API token and Channel ID from Windmill variables.
    """
    try:
        slack_token = wmill.get_variable("u/niccolosalvini27/SLACK_API_TOKEN")
        slack_channel = wmill.get_variable("u/niccolosalvini27/SLACK_CHANNEL_ID")

        if not slack_token or not slack_channel:
            print("Slack environment variables not found (SLACK_API_TOKEN, SLACK_CHANNEL_ID). Notification skipped.")
            return

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {slack_token}",
            "Content-type": "application/json; charset=utf-8"
        }
        payload = {
            "channel": slack_channel,
            "text": message_text
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        response_json = response.json()
        if response_json.get("ok"):
            print("Summary notification sent to Slack successfully.")
        else:
            print(f"Error sending to Slack: {response_json.get('error')}")

    except requests.exceptions.RequestException as e:
        print(f"Network error during Slack notification: {e}")
    except Exception as e:
        print(f"Unexpected error during Slack notification: {e}")

# ============================================================
# S3 DATA PROCESSOR FOR MINIO STORAGE
# ============================================================

class S3DataProcessor:
    """Handles reading from and writing to S3-compatible storage (MinIO)"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("OutliersProcessor")
        
        # Initialize S3 client
        self.s3_client = self._setup_s3_client()
        self.s3_bucket = self._get_s3_bucket()
        
    def _setup_s3_client(self):
        """Setup S3 client with credentials from Windmill"""
        try:
            # Get credentials from Windmill
            s3_endpoint_url = wmill.get_variable("u/niccolosalvini27/S3_ENDPOINT_URL")
            s3_access_key = wmill.get_variable("u/niccolosalvini27/S3_ACCESS_KEY")
            s3_secret_key = wmill.get_variable("u/niccolosalvini27/S3_SECRET_KEY")

            self.logger.info(f"S3 Configuration: endpoint={s3_endpoint_url}, access_key={s3_access_key[:8]}...")

            # Create S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=s3_endpoint_url,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                region_name='us-east-1'  # MinIO default
            )
            
            # Test the connection by listing buckets
            try:
                buckets_response = s3_client.list_buckets()
                self.logger.info(f"S3 connection successful. Available buckets: {[b['Name'] for b in buckets_response['Buckets']]}")
            except Exception as e:
                self.logger.warning(f"S3 connection test failed: {e}")
            
            self.logger.info(f"S3 client initialized with endpoint: {s3_endpoint_url}")
            return s3_client
            
        except Exception as e:
            self.logger.error(f"Failed to setup S3 client: {e}")
            raise
    
    def _get_s3_bucket(self):
        """Get S3 bucket name from Windmill variables"""
        try:
            bucket_name = wmill.get_variable("u/niccolosalvini27/S3_BUCKET")
            if not bucket_name:
                raise ValueError("S3_BUCKET variable not found in Windmill")
            return bucket_name
        except Exception as e:
            self.logger.error(f"Failed to get S3 bucket name: {e}")
            raise
    
    def list_symbol_files(self, symbol, asset_type, file_format='parquet'):
        """List all data files for a symbol in S3, handling pagination"""
        try:
            # S3 prefix for the symbol
            prefix = f"data/{asset_type}/{symbol}/"
            
            files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
            
            extension = '.parquet' if file_format == 'parquet' else '.txt'
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        filename = key.split('/')[-1]
                        
                        # Filter files based on format and exclude special files
                        if filename.endswith(extension):
                            if filename not in ['adjustment.txt', f'{symbol}_last_update.txt']:
                                files.append(filename)
            
            self.logger.debug(f"Found {len(files)} files for {symbol}")
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"Error listing files for {symbol}: {e}")
            return []
    
    def read_data_file(self, symbol, asset_type, filename, file_format='parquet'):
        """Read a data file from S3"""
        try:
            s3_key = f"data/{asset_type}/{symbol}/{filename}"
            
            # Download file content
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            file_content = response['Body'].read()
            
            if file_format == 'parquet':
                # Handle parquet files
                try:
                    
                    df = pq.read_table(io.BytesIO(file_content)).to_pandas()
                    # Ensure column names are consistent
                    if 'time' not in df.columns and len(df.columns) >= 4:
                        df.columns = ['time', 'price', 'volume', 'trades'] + list(df.columns[4:])
                except ImportError:
                    self.logger.warning(f"PyArrow not available, attempting CSV read for {filename}")
                    # Fallback to CSV reading
                    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), header=None)
            else:  # txt format
                df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), header=None, 
                               dtype={0: 'str', 1: 'float', 2: 'int', 3: 'int'})
                df.columns = ['time', 'price', 'volume', 'trades']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading {symbol}/{filename}: {e}")
            return None
    
    def write_data_file(self, df, symbol, asset_type, filename, file_format='parquet'):
        """Write a data file to S3"""
        try:
            s3_key = f"data/{asset_type}/{symbol}/{filename}"
            
            if file_format == 'parquet':
                
                parquet_buffer = io.BytesIO()
                df.columns = [str(col) for col in df.columns]
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_buffer, compression='snappy')
                parquet_buffer.seek(0)
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=parquet_buffer.getvalue(),
                    ContentType='application/octet-stream'
                )
            else:  # txt format
                # Save as CSV
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, header=False, sep=',')
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=csv_buffer.getvalue(),
                    ContentType='text/csv'
                )
            
            self.logger.debug(f"Successfully wrote {symbol}/{filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing {symbol}/{filename}: {e}")
            return False
    
    def get_last_update_info(self, symbol, asset_type):
        """Get last update info from S3"""
        try:
            s3_key = f"data/{asset_type}/{symbol}/{symbol}_last_update.txt"
            
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            
            # Parse the last line to get start date
            lines = content.strip().split('\n')
            if lines:
                last_line = lines[-1]
                # Split the row and take the second column (start_date)
                parts = last_line.split(',')
                if len(parts) >= 2:
                    second_column = parts[1].strip()
                    # Return the date in yyyy_mm_dd format
                    return second_column
            
            return None
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.debug(f"No last update file found for {symbol}")
                return None
            else:
                self.logger.error(f"Error reading last update info for {symbol}: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Error reading last update info for {symbol}: {e}")
            return None

# ============================================================
# EXACT COPY OF ORIGINAL OUTLIERS_DETECTION.PY FUNCTIONS
# ============================================================

def setup_logger_console():
    """Setup logger for outliers processing - CONSOLE OUTPUT for Windmill"""
    # Configure logger for console output only
    logger = logging.getLogger('outliers_processor')

    # IF THE LOGGER ALREADY HAS HANDLERS, DO NOT RECONFIGURE IT
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    # AVOID DUPLICATES
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler only (no file handler for Windmill)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

@njit
def trimmed_mean_and_std(x, delta):
    trim_count = np.floor(len(x) * (delta / 2))
    sorted_x = np.sort(x)
    if trim_count == 0:
       trimmed_x = sorted_x
       print('no trimming on the window', len(trimmed_x))
    else:
       trimmed_x = sorted_x[trim_count:-trim_count]
       #print(len(trimmed_x))
    mean = np.mean(trimmed_x)
    std = np.sqrt(np.sum((trimmed_x - mean) ** 2) / (len(trimmed_x) - 1))  # Manually calculate std with ddof=1
    return mean, std

@njit 
def k_neighborhood_excluding_i(prices, i, k):
    n = len(prices)
    half_k = k // 2
    
    if i < half_k:
        neighborhood = prices[:k + 1]
        neighborhood = np.delete(neighborhood, i)
    elif i >= n - half_k:
        neighborhood = prices[-(k + 1):]
        neighborhood = np.delete(neighborhood, i-n)
    else:
        neighborhood = prices[i - half_k:i + half_k+1]
        neighborhood = np.delete(neighborhood, half_k)
    
    return neighborhood

@njit#(parallel=True)
def detect_outliers_numba(prices):

    n = len(prices)
    delta =0.1
    gamma = 0.06
    k = 120

    outliers = []
    #result = np.copy(prices)
    
    for i in prange(n):
        neighborhood = k_neighborhood_excluding_i(prices, i, k)
        trimmed_mean_neighborhood, trimmed_std_neighborhood = trimmed_mean_and_std(neighborhood, delta)
        if abs(prices[i] - trimmed_mean_neighborhood) >= (3 * trimmed_std_neighborhood + gamma):
            #outliers.append(i-1) # Se input ret
            outliers.append(i) # Se input prices
            #result[i] = np.nan 
    
    return outliers

def prepare_data_from_df(df, file_format='parquet'):
    """Prepare data from DataFrame instead of file path - ADAPTED FOR S3"""
    
    # Ensure column names are consistent
    if file_format.lower() == 'parquet':
        # For parquet, ensure column names are consistent with txt format
        if 'time' not in df.columns and len(df.columns) >= 4:
            df.columns = ['time', 'price', 'volume', 'trades'] + list(df.columns[4:])
    else:  # Default to txt
        if len(df.columns) >= 4:
            df.columns = ['time', 'price', 'volume', 'trades'] + list(df.columns[4:])

    # Funzione migliorata per aggiungere millisecondi solo se necessario
    def add_milliseconds(time_str):
        # Convert to string if not already
        time_str = str(time_str)
        # Add milliseconds only if not already present
        if '.' not in time_str:
            return time_str + '.000'
        return time_str
    
    # Only apply the function if the 'time' column contains strings
    if pd.api.types.is_string_dtype(df['time']):
        df['time'] = df['time'].apply(add_milliseconds)
        
    return df

def filter_trading_hours(df, day_to_check, early_closing_day_file=None):
    """
    Filter DataFrame based on trading hours, with special handling for early closing days.
    ADAPTED FOR S3 - early_closing_day_file is now optional since we get it from Windmill
    """
    # Get the day from the file name by removing the extension
    day = day_to_check.replace('.txt', '').replace('.parquet', '')

    # Get early closing days from Windmill variable instead of file
    try:
        early_closing_days_str = wmill.get_variable("u/niccolosalvini27/EARLY_CLOSING_DAYS")
        if early_closing_days_str:
            early_closing_days = [line.strip() for line in early_closing_days_str.split('\n') if line.strip()]
        else:
            early_closing_days = []
    except Exception as e:
        print(f"Error getting early closing days from Windmill: {e}")
        early_closing_days = []
    
    # Determine closing time by day
    if day in early_closing_days:
        end_time = datetime.strptime('12:59:59.999', '%H:%M:%S.%f').time()
    else:
        end_time = datetime.strptime('15:59:59.999', '%H:%M:%S.%f').time()
    
    # Standard opening hours
    start_time = datetime.strptime('09:30:00.000', '%H:%M:%S.%f').time()
    
    # Creates a copy to avoid modifying the original DataFrame
    filtered_df = df.copy()
    
    # Convert the time column to datetime if it is not already
    if 'time' in filtered_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['time']):
            try:
                # Convert to datetime
                filtered_df['time'] = pd.to_datetime(filtered_df['time'], format='mixed', errors='coerce')
            except Exception as e:
                print(f"Error converting time to datetime: {e}")
                return filtered_df  # Returns unfiltered DataFrame in case of error
        
        # Extract only the hourly part
        time_only = filtered_df['time'].dt.time
        
        # Filter the DataFrame
        filtered_df = filtered_df[
            (time_only >= start_time) & 
            (time_only <= end_time)
        ]
    
    return filtered_df

def add_outliers_column_s3(df, filtered_df, s3_processor, symbol, asset_type, filename, file_format='parquet'):
    """Add outliers column and save to S3 - ADAPTED FOR S3 STORAGE"""
    prices = filtered_df['price'].values
    outliers = detect_outliers_numba(prices)
    #print("Outliers trovati:", len(outliers))

    # Converts positions into absolute indices
    outlier_indices = filtered_df.index[outliers]

    # Set is_not_outlier to NaN where needed
    filtered_df['is_not_outlier'] = np.where(filtered_df.index.isin(outlier_indices), np.nan, 1)

    #print("NaN in filtered_df['is_not_outlier']:", filtered_df['is_not_outlier'].isna().sum())

    # Initialise everything to NaN in the original df
    df['is_not_outlier'] = 0
    df.loc[filtered_df.index, 'is_not_outlier'] = filtered_df['is_not_outlier']

    #print("NaN in df['is_not_outlier']:", df['is_not_outlier'].isna().sum())

    # Save file to S3 instead of local disk
    success = s3_processor.write_data_file(df, symbol, asset_type, filename, file_format)
    
    return df if success else None

# Thread-local storage for thread safety
thread_local = threading.local()

def process_all_symbols_threaded_s3(config, s3_processor, max_workers=None):
    """
    Process all symbols using ThreadPoolExecutor for improved performance - S3 VERSION
    """
    # Setup logger
    logger = setup_logger_console()
    logger.info(f"Starting outliers processing for symbols: {config['symbols']}")
    logger.info(f"File format: {config['file_format']}, Asset type: {config.get('asset_type', 'stocks')}")

    # Process symbols in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each symbol for processing
        futures = [executor.submit(process_single_symbol_s3, config, symbol, s3_processor) 
                  for symbol in config['symbols']]
        
        # Wait for all futures to complete
        for future in futures:
            try:
                future.result()  # This will re-raise any exception that occurred
            except Exception as e:
                print(f"Error processing symbol: {e}")
    
    print("All symbols processed")

def process_single_symbol_s3(config, symbol, s3_processor):
    """Process a single symbol with all its files - S3 VERSION"""
    try:
        # Get start date from S3 instead of local file
        asset_type = config.get('asset_type', 'stocks')  # Default to stocks
        start_date = s3_processor.get_last_update_info(symbol, asset_type)
        
        # Create a thread-local copy of config to avoid contention
        if not hasattr(thread_local, 'config'):
            thread_local.config = {}
        
        # Copy relevant parts of config_out to thread_local.config
        thread_local.config = config.copy()
        thread_local.config['start_date'] = start_date

        logger = logging.getLogger('outliers_processor')
        logger.info(f"Symbol {symbol} - Start date: {start_date}")
        
        # Initialize thread-local functions
        initialize_thread_local_functions_s3()
        
        # Process files for this symbol
        process_symbol_files_threaded_s3(thread_local.config, symbol, s3_processor)
        
    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        raise  # Re-raise to be caught by the executor

def process_symbol_files_threaded_s3(config, symbol, s3_processor):
    """Thread-safe version of process_symbol_files - S3 VERSION"""

    asset_type = config.get('asset_type', 'stocks')  # Default to stocks
    logger = logging.getLogger('outliers_processor')

    # Get list of files from S3 instead of local directory
    file_format = config.get('file_format', 'parquet').lower()
    all_files = s3_processor.list_symbol_files(symbol, asset_type, file_format)
    
    if not all_files:
        logger.info(f"No files found for {symbol} in S3")
        return

    # Filter by start_date if provided
    if 'start_date' in config and config['start_date']:
        start_date = config['start_date']
        file_extension = '.parquet' if file_format == 'parquet' else '.txt'
        # Extract date part from filename (removing extension)
        all_files = [f for f in all_files if f.replace(file_extension, '') >= start_date]
            
    # Sort files by date
    files = sorted(all_files)
    
    if not files:
        logger.info(f"No files to process for {symbol}")
        return
    
    try:
        initialize_thread_local_functions_s3()

        # Process each file
        processed = 0
        skipped = 0
        errors = 0
        
        for file in files:
            result = process_single_file_s3(symbol, asset_type, file, config.get('file_format', 'parquet'), s3_processor)
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            else:  # "error"
                errors += 1
        
        logger.info(f"Symbol {symbol} completed: {processed} processed, {skipped} skipped, {errors} errors")
        
    except Exception as e:
        logger.error(f"Error processing files for {symbol}: {e}")
        raise

def initialize_thread_local_functions_s3():
    """Initialize thread-local versions of processing functions - S3 VERSION"""
    if not hasattr(thread_local, 'functions_initialized'):
        thread_local.prepare_data_from_df = prepare_data_from_df
        thread_local.filter_trading_hours = filter_trading_hours
        thread_local.add_outliers_column_s3 = add_outliers_column_s3
        thread_local.functions_initialized = True

def process_single_file_s3(symbol, asset_type, filename, file_format, s3_processor):
    """Process a single file and return the result status - S3 VERSION"""
    
    initialize_thread_local_functions_s3()
    
    try:
        # Read file from S3
        df = s3_processor.read_data_file(symbol, asset_type, filename, file_format)
        
        if df is None:
            return "error"
        
        # Check columns - skip if already has 5 columns (already processed)
        if len(df.columns) == 5:
            print(f"Skipping {symbol}/{filename} - has 5 columns")
            return "skipped"
        
        # Process the file
        data = thread_local.prepare_data_from_df(df, file_format)
        filtered_df = thread_local.filter_trading_hours(data, filename)
        result_df = thread_local.add_outliers_column_s3(
            data, filtered_df, s3_processor, symbol, asset_type, filename, file_format
        )
        
        if result_df is not None:
            return "processed"
        else:
            return "error"
            
    except Exception as e:
        print(f"Error processing {symbol}/{filename}: {e}")
        return "error"

# ============================================================
# MAIN FUNCTION FOR WINDMILL
# ============================================================

def main(
    # Pipeline configuration
    pipeline_name="outliers_detection",
    pipeline_enabled=True,
    
    # Asset type configuration - ONLY stocks and ETFs get outlier detection
    stocks_enabled=True,
    stocks_symbols=["MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY", "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA", "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM", "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK", "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC"],
    
    etfs_enabled=True,
    etfs_symbols=["AGG", "BND", "GLD", "SLV", "SUSA", "EFIV", "ESGV", "ESGU", "AFTY", "MCHI", "EWH", "EEM", "IEUR", "VGK", "FLCH", "EWJ", "NKY", "EWZ", "EWC", "EWU", "EWI", "EWP", "ACWI", "IOO", "GWL", "VEU", "IJH", "MDY", "IVOO", "IYT", "XTN", "XLI", "XLU", "VPU", "SPSM", "IJR", "VIOO", "QQQ", "ICLN", "ARKK", "SPLG", "SPY", "VOO", "IYY", "VTI", "DIA"],
    
    # NOTE: forex and futures are NOT included because they don't get outlier detection
    # This follows the original logic: if asset_type.lower() in ['stocks', 'etfs'] and 'outliers' in steps_to_run
    
    # Processing settings
    file_format="parquet",
    outliers_threads_max=4,
    verbose=False
):
    """
    Windmill-Compatible Outliers Detection
    
    Preserves IDENTICAL logic to original outliers_detection.py but with:
    - S3 storage integration instead of local files
    - Console logging instead of file logging
    - Windmill variable integration
    - Slack notifications
    
    IMPORTANT: Only processes stocks and ETFs as per original logic.
    Forex and futures do NOT get outlier detection.
    """
    start_time = datetime.now()
    
    # Setup console logging
    print(f"OUTLIERS DETECTION START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Build enabled pipelines list - ONLY stocks and ETFs
        enabled_pipelines = []
        
        if stocks_enabled:
            enabled_pipelines.append({
                'name': 'stocks',
                'asset_type': 'stocks',
                'symbols': stocks_symbols,
                'steps': ['outliers']
            })
        
        if etfs_enabled:
            enabled_pipelines.append({
                'name': 'ETFs',
                'asset_type': 'ETFs',
                'symbols': etfs_symbols,
                'steps': ['outliers']
            })
        
        if not enabled_pipelines:
            print("No pipelines enabled for outlier detection. Note: Only stocks and ETFs get outlier detection.")
            return
        
        print(f"Enabled pipelines for outlier detection: {[p['name'] for p in enabled_pipelines]}")
        print("NOTE: Forex and futures are excluded from outlier detection as per original logic.")
        
        # Process each pipeline
        all_results = []
        total_processed = 0
        total_errors = 0
        total_skipped = 0
        
        for pipeline in enabled_pipelines:
            print(f"\n--- Processing outliers for pipeline: {pipeline['name']} ({pipeline['asset_type']}) ---")
            
            # Create config for this pipeline
            config = {
                'asset_type': pipeline['asset_type'],
                'symbols': pipeline['symbols'],
                'file_format': file_format,
                'outliers_threads_max': outliers_threads_max
            }
            
            # Initialize S3 processor
            s3_processor = S3DataProcessor(config)
            
            print(f"Processing {len(pipeline['symbols'])} symbols for {pipeline['asset_type']} outlier detection...")
            
            # Process symbols with IDENTICAL logic
            process_all_symbols_threaded_s3(config, s3_processor, max_workers=outliers_threads_max)
            
            # For now, we'll track basic success (could be enhanced with detailed stats)
            pipeline_result = {
                'pipeline': pipeline['name'],
                'asset_type': pipeline['asset_type'],
                'symbols_count': len(pipeline['symbols']),
                'success': True  # Basic success tracking
            }
            all_results.append(pipeline_result)
            
            print(f"Pipeline {pipeline['name']} completed successfully")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        successful_pipelines = [r for r in all_results if r['success']]
        success_rate = len(successful_pipelines) / len(all_results) * 100 if all_results else 0
        
        print("\n====== FINAL OUTLIERS DETECTION SUMMARY ======")
        print(f"Total execution time: {duration}")
        print(f"Pipelines processed: {len(all_results)}")
        print(f"Successful pipelines: {len(successful_pipelines)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total symbols processed: {sum(r['symbols_count'] for r in all_results)}")
        
        # Determine status
        if len(successful_pipelines) == len(all_results):
            print("ALL OUTLIER DETECTION COMPLETED SUCCESSFULLY!")
            status_emoji = "✅"
            status_text = "SUCCESS"
        else:
            print("SOME OUTLIER DETECTION ERRORS OCCURRED!")
            status_emoji = "⚠️" if len(successful_pipelines) > 0 else "❌"
            status_text = "PARTIAL SUCCESS" if len(successful_pipelines) > 0 else "FAILURE"
        
        # Create Slack message
        slack_message = f"{status_emoji} *OUTLIERS DETECTION COMPLETED* {status_emoji}\n\n"
        slack_message += f"*Status:* {status_text}\n"
        slack_message += f"*Pipeline:* {pipeline_name}\n"
        slack_message += f"*Duration:* {str(duration).split('.')[0]}\n"
        slack_message += f"*Pipelines:* {len(successful_pipelines)}/{len(all_results)} successful\n\n"
        
        # Add pipeline details
        slack_message += "*Pipeline Results:*\n"
        for result in all_results:
            status_icon = "✅" if result['success'] else "❌"
            slack_message += f"{status_icon} {result['pipeline']} ({result['asset_type']}): {result['symbols_count']} symbols\n"
        
        slack_message += f"\n*Processing Summary:*"
        slack_message += f"\n📊 Total symbols: {sum(r['symbols_count'] for r in all_results)}"
        slack_message += f"\n🎯 Asset types: {', '.join(set(r['asset_type'] for r in all_results))}"
        slack_message += f"\n📈 Success rate: {success_rate:.1f}%"
        
        slack_message += f"\n\n*Note:* Only stocks and ETFs processed (forex/futures excluded as per original logic)"
        slack_message += f"\n*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send Slack notification
        try:
            send_slack_notification(slack_message)
        except Exception as e:
            print(f"Error sending Slack notification: {e}")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        logger.error(f"CRITICAL ERROR: {e}", exc_info=True)
        
        # Send error notification to Slack
        try:
            error_message = f"❌ *OUTLIERS DETECTION FAILED* ❌\n\n"
            error_message += f"*Error:* {str(e)}\n"
            error_message += f"*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            send_slack_notification(error_message)
        except Exception as slack_error:
            print(f"Failed to send error notification: {slack_error}")
