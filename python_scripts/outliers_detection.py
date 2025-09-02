"""
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics. 
Research code - Unauthorized distribution prohibited.
"""

# B. G. Procedure for tickms data - 15-05-2025
# process all file in txt or parquet in the folder
# Functions
import numpy as np
import pandas as pd
from io import BytesIO
from numba import njit, prange
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os 
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# ====================================================================
# Clean for outliers BG
# ====================================================================


def setup_logger(base_dir):
    """Setup logger for outliers processing"""
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.now().strftime("%Y_%m_%d")
    log_file = os.path.join(log_dir, f'outliers_{today}.log')
    
    # Configure logger
    logger = logging.getLogger('outliers_processor')

    # IF THE LOGGER ALREADY HAS HANDLERS, DO NOT RECONFIGURE IT
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    # AVOID DUPLICATES
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
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

def prepare_data(file_path, file_format='txt'):
     
    if file_format.lower() == 'parquet':
        # Read parquet file
        df = pd.read_parquet(file_path)
        # Ensure column names are consistent with txt format
        if 'time' not in df.columns and len(df.columns) >= 4:
            df.columns = ['time', 'price', 'volume', 'trades'] + list(df.columns[4:])
    else:  # Default to txt
        df = pd.read_csv(file_path, header=None, dtype={0: 'str', 1: 'float', 2: 'int', 3: 'int'})
        df.columns = ['time', 'price', 'volume', 'trades']


    # Funzione migliorata per aggiungere millisecondi solo se necessario
    def add_milliseconds(time_str):
        # Convert to string if not already
        time_str = str(time_str)
        # Add milliseconds only if not already present
        if '.' not in time_str:
            return time_str + '.000'
        return time_str
    
    # Only apply the function if the ‘time’ column contains strings
    if pd.api.types.is_string_dtype(df['time']):
        df['time'] = df['time'].apply(add_milliseconds)
        
    return df

def filter_trading_hours(df, day_to_check, early_closing_day_file):
    """
    Filter DataFrame based on trading hours, with special handling for early closing days.
    """
    # Get the day from the file name by removing the extension
    day = day_to_check.replace('.txt', '').replace('.parquet', '')

    # Read the file of early closing days
    try:
        with open(early_closing_day_file, 'r') as f:
            early_closing_days = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"No early closing days file found at {early_closing_day_file}")
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

def add_outliers_column(df, filtered_df, file_path, file_format='txt'):
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

    df.to_csv(file_path, index=False, header=False, sep=',')

    # Save file in the specified format
    if file_format.lower() == 'parquet':          
        df.to_parquet(file_path, index=False)
    else:  # Default to txt
        df.to_csv(file_path, index=False, header=False, sep=',')

    return df

def get_start_date_from_log(log_file_path):
    with open(log_file_path, 'r') as log_file:
        # Read all lines of the file
        lines = log_file.readlines()
        
        # If the file is not empty
        if lines:
            # Take the last line (last line of downloading done)
            last_line = lines[-1]
            # Split the row and take the second column (start_date)
            second_column = last_line.split(',')[1].strip()  # Prende la seconda colonna dell'ultima riga
            # Set the date in yyyy-mm-dd format
            start_date = datetime.strptime(second_column, "%Y_%m_%d").strftime("%Y_%m_%d")
            return start_date
        else:
            # If the file is empty, return None
            return None

# Thread-local storage for thread safety
thread_local = threading.local()

def process_all_symbols_threaded(config, max_workers=None):
    """
    Process all symbols using ThreadPoolExecutor for improved performance.
    """
    # Setup logger
    logger = setup_logger(config['base_dir'])
    logger.info(f"Starting outliers processing for symbols: {config['symbols']}")
    logger.info(f"File format: {config['file_format']}, Asset type: {config.get('asset_type', 'stocks')}")

    # Process symbols in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each symbol for processing
        futures = [executor.submit(process_single_symbol, config, symbol) 
                  for symbol in config['symbols']]
        
        # Wait for all futures to complete
        for future in futures:
            try:
                future.result()  # This will re-raise any exception that occurred
            except Exception as e:
                print(f"Error processing symbol: {e}")
    
    print("All symbols processed")

def process_single_symbol(config, symbol):
    """Process a single symbol with all its files."""
    try:
        # Get start date from the symbol's last_update file
        # Get start date from the symbol's last_update file (stocks/etf only)
        asset_type = config.get('asset_type', 'stocks')  # Default to stocks
        info_file_path = os.path.join(config['base_dir'], asset_type, symbol, f'{symbol}_last_update.txt')
        start_date = get_start_date_from_log(info_file_path)
        
        # Create a thread-local copy of config to avoid contention
        if not hasattr(thread_local, 'config'):
            thread_local.config = {}
        
        # Copy relevant parts of config_out to thread_local.config
        thread_local.config = config.copy()
        thread_local.config['start_date'] = start_date

        # Make sure early_closing_day_file is in the thread-local config
        if 'early_closing_day_file' not in thread_local.config and 'early_closing_day_file' in config:
            thread_local.config['early_closing_day_file'] = config['early_closing_day_file']
        
        #print(f"Symbol {symbol} - Start date: {start_date}")
        logger = logging.getLogger('outliers_processor')
        logger.info(f"Symbol {symbol} - Start date: {start_date}")
        
        # Initialize thread-local functions
        initialize_thread_local_functions()
        
        # Process files for this symbol
        process_symbol_files_threaded(thread_local.config, symbol)
        
    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        raise  # Re-raise to be caught by the executor

def process_symbol_files_threaded(config, symbol):
    """Thread-safe version of process_symbol_files."""

    # Construct the symbol directory path
    asset_type = config.get('asset_type', 'stocks')  # Default to stocks
    symbol_dir = os.path.join(config['base_dir'], asset_type, symbol)
    logger = logging.getLogger('outliers_processor')

    # Check if directory exists
    if not os.path.exists(symbol_dir):
        logger.warning(f"Directory for {symbol} does not exist: {symbol_dir}")
        return
    
    # Get list of files to process based on file format
    file_format = config.get('file_format', 'txt').lower()

    # File extension to look for
    file_extension = '.parquet' if file_format == 'parquet' else '.txt'

    # Excluded files should only be considered for txt format
    excluded_files = []
    if file_format == 'txt':
        excluded_files = ['adjustment.txt', f'{symbol}_last_update.txt']
    
    try:
        initialize_thread_local_functions()

        # Get only files that match the specified format
        all_files = [f for f in os.listdir(symbol_dir) 
                    if f.endswith(file_extension)]
        
        # For txt format, exclude specific files
        if file_format == 'txt':
            all_files = [f for f in all_files if f not in excluded_files]

        # Filter by start_date if provided
        if 'start_date' in config and config['start_date']:
            start_date = config['start_date']
            # Extract date part from filename (removing extension)
            all_files = [f for f in all_files if f.replace(file_extension, '') >= start_date]
                
        # Sort files by date
        files = sorted(all_files)
        
        if not files:
            logger.info(f"No files to process for {symbol}")
            return
        
        # Process each file
        processed = 0
        skipped = 0
        errors = 0
        
        for file in files:
            result = process_single_file(symbol_dir, symbol, file, config.get('file_format', 'txt'))
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            else:  # "error"
                errors += 1
        
        logger.info(f"Symbol {symbol} completed: {processed} processed, {skipped} skipped, {errors} errors")
        
    except Exception as e:
        logger.info(f"Error processing files for {symbol}: {e}")
        raise

def initialize_thread_local_functions():
    """Initialize thread-local versions of processing functions."""
    if not hasattr(thread_local, 'functions_initialized'):
        thread_local.prepare_data = prepare_data
        thread_local.filter_trading_hours = filter_trading_hours
        thread_local.add_outliers_column = add_outliers_column
        thread_local.functions_initialized = True

def process_single_file(symbol_dir, symbol, file, file_format='txt'):
    """Process a single file and return the result status."""
    file_path = os.path.join(symbol_dir, file)

    initialize_thread_local_functions()
    
    try:
        # Check columns based on file format
        if file_format == 'txt':
            # Check the number of columns quickly without loading the entire file
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    column_count = len(first_line.split(','))
                    if column_count == 5:
                        print(f"Skipping {symbol}/{file} - has 5 columns")
                        return "skipped"
        elif file_format == 'parquet':
            # For parquet, we need to read metadata
            try:
                df_sample = pd.read_parquet(file_path, columns=[])
                if len(df_sample.columns) == 5:
                    print(f"Skipping {symbol}/{file} - has 5 columns")
                    return "skipped"
            except Exception as e:
                print(f"Error checking parquet columns in {symbol}/{file}: {e}")
                return "error"
    except Exception as e:
        print(f"Error checking columns in {symbol}/{file}: {e}")
        return "error"
    
    # Process the file
    try:    
        # Use thread-local functions to avoid contention
        data = thread_local.prepare_data(file_path, file_format)
        filtered_df = thread_local.filter_trading_hours(data, file, thread_local.config['early_closing_day_file'])
        df = thread_local.add_outliers_column(data, filtered_df, file_path, file_format)
        
        return "processed"
            
    except Exception as e:
        print(f"Error processing {symbol}/{file}: {e}")
        return "error"

