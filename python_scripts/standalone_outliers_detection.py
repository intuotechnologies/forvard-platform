"""
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics. 
Research code - Unauthorized distribution prohibited.

This script is a standalone outlier detector for financial data. It reads data from
an S3-compatible datalake, processes outliers using the BG procedure, and saves
the results back to the same datalake.
"""

import os
import time
import logging
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from numba import njit, prange
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys
from collections import defaultdict
import wmill
import s3fs
import pyarrow


# --- Globals ---
print_lock = threading.Lock()
thread_local = threading.local()

# --- Logging and printing functions ---
def format_log_header(type, message):
    '''Format a header to make certain types of messages more visible'''
    if type == "PHASE":
        return f"\\n{'='*30} {message} {'='*30}"
    elif type == "PROCESS":
        return f"\\n[PROCESS] {'-'*10} {message} {'-'*10}"
    elif type == "STEP":
        return f"\\n  [STEP] {message}"
    elif type == "COMPLETE":
        return f"\\n[COMPLETE] {'-'*10} {message} {'-'*10}"
    elif type == "INFO":
        return f"[INFO] {message}"
    elif type == "WARNING":
        return f"[WARNING] {message}"
    elif type == "ERROR":
        return f"[ERROR] {message}"
    else:
        return f"[{type}] {message}"

def safe_print(message, log_type=None):
    '''Thread-safe printing with improved formatting'''
    with print_lock:
        if log_type:
            formatted_message = format_log_header(log_type, message)
            print(formatted_message)
            logging.info(formatted_message)
        else:
            print(message)
            logging.info(message)

# Logger configuration
logger = logging.getLogger("OutliersProcessor")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ====================================================================
# Outlier Detection Functions - IDENTICAL TO ORIGINAL
# ====================================================================

@njit
def trimmed_mean_and_std(x, delta):
    trim_count = np.floor(len(x) * (delta / 2))
    sorted_x = np.sort(x)
    if trim_count == 0:
       trimmed_x = sorted_x
    else:
       trimmed_x = sorted_x[trim_count:-trim_count]
    mean = np.mean(trimmed_x)
    std = np.sqrt(np.sum((trimmed_x - mean) ** 2) / (len(trimmed_x) - 1))
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

@njit
def detect_outliers_numba(prices):
    n = len(prices)
    delta = 0.1
    gamma = 0.06
    k = 120

    outliers = []
    
    for i in prange(n):
        neighborhood = k_neighborhood_excluding_i(prices, i, k)
        trimmed_mean_neighborhood, trimmed_std_neighborhood = trimmed_mean_and_std(neighborhood, delta)
        if abs(prices[i] - trimmed_mean_neighborhood) >= (3 * trimmed_std_neighborhood + gamma):
            outliers.append(i)
    
    return outliers

# ====================================================================
# S3 Data Processing Functions
# ====================================================================

class S3DataProcessor:
    """Handles reading from and writing to S3-compatible storage"""
    
    def __init__(self):
        # S3 Configuration from Windmill/Env
        self.s3_endpoint_url = wmill.get_variable("u/niccolosalvini27/S3_ENDPOINT_URL")
        self.s3_access_key = wmill.get_variable("u/niccolosalvini27/S3_ACCESS_KEY")
        self.s3_secret_key = wmill.get_variable("u/niccolosalvini27/S3_SECRET_KEY")
        self.s3_bucket = wmill.get_variable("u/niccolosalvini27/S3_BUCKET_DEV")
        
        # Check for S3 configuration
        if not all([self.s3_endpoint_url, self.s3_access_key, self.s3_secret_key, self.s3_bucket]):
            error_msg = "S3 environment variables are not set. Cannot access datalake."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not s3fs:
            error_msg = "The 's3fs' package is required to access S3."
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Initialize S3 filesystem
        self.s3 = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': self.s3_endpoint_url},
            key=self.s3_access_key,
            secret=self.s3_secret_key
        )
    
    def list_symbol_files(self, symbol, asset_type, file_format='parquet'):
        """List all files for a symbol in the S3 bucket"""
        try:
            # Construct the S3 path: data/asset_type/symbol/
            s3_path = f"{self.s3_bucket}/data/{asset_type}/{symbol}/"
            
            # Get list of files
            files = self.s3.ls(s3_path, detail=False)
            
            # Filter files
            extension = '.parquet' if file_format == 'parquet' else '.txt'
            symbol_files = []
            
            for file_path in files:
                filename = os.path.basename(file_path)
                if filename.endswith(extension):
                    # Exclude special files for txt format
                    if file_format == 'txt' and filename in ['adjustment.txt', f'{symbol}_last_update.txt']:
                        continue
                    symbol_files.append(filename)
            
            return sorted(symbol_files)
            
        except Exception as e:
            logger.error(f"Error listing files for {symbol}: {e}")
            return []
    
    def read_data_file(self, symbol, asset_type, filename, file_format='parquet'):
        """Read a data file from S3"""
        try:
            s3_path = f"{self.s3_bucket}/data/{asset_type}/{symbol}/{filename}"
            
            if file_format == 'parquet':
                with self.s3.open(s3_path, 'rb') as f:
                    df = pd.read_parquet(f)
                    # Ensure column names are consistent
                    if 'time' not in df.columns and len(df.columns) >= 4:
                        df.columns = ['time', 'price', 'volume', 'trades'] + list(df.columns[4:])
            else:  # txt format
                with self.s3.open(s3_path, 'r') as f:
                    df = pd.read_csv(f, header=None, dtype={0: 'str', 1: 'float', 2: 'int', 3: 'int'})
                    df.columns = ['time', 'price', 'volume', 'trades']
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading {symbol}/{filename}: {e}")
            return None
    
    def write_data_file(self, df, symbol, asset_type, filename, file_format='parquet'):
        """Write a data file to S3"""
        try:
            s3_path = f"{self.s3_bucket}/data/{asset_type}/{symbol}/{filename}"
            
            if file_format == 'parquet':
                with self.s3.open(s3_path, 'wb') as f:
                    df.to_parquet(f, index=False)
            else:  # txt format
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, header=False, sep=',')
                with self.s3.open(s3_path, 'w') as f:
                    f.write(csv_buffer.getvalue())
            
            logger.debug(f"Successfully wrote {symbol}/{filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing {symbol}/{filename}: {e}")
            return False
    
    def get_last_update_info(self, symbol, asset_type):
        """Get start date from last update file"""
        try:
            s3_path = f"{self.s3_bucket}/data/{asset_type}/{symbol}/{symbol}_last_update.txt"
            
            with self.s3.open(s3_path, 'r') as f:
                lines = f.readlines()
                
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    parts = last_line.split(',')
                    if len(parts) >= 2:
                        second_column = parts[1].strip()
                        try:
                            start_date = datetime.strptime(second_column, "%Y_%m_%d").strftime("%Y_%m_%d")
                            return start_date
                        except ValueError:
                            pass
            return None
            
        except Exception:
            return None

# ====================================================================
# Data Processing Functions - IDENTICAL LOGIC TO ORIGINAL
# ====================================================================

def prepare_data(df, file_format='parquet'):
    """Prepare data - identical logic to original"""
    def add_milliseconds(time_str):
        time_str = str(time_str)
        if '.' not in time_str:
            return time_str + '.000'
        return time_str
    
    if pd.api.types.is_string_dtype(df['time']):
        df['time'] = df['time'].apply(add_milliseconds)
        
    return df

def filter_trading_hours(df, day_to_check):
    """Filter DataFrame based on trading hours - identical logic to original"""
    # Hardcoded early closing days (from config)
    early_closing_days = [
        '2015_11_27', '2015_12_24', '2016_11_25', '2016_12_24', '2017_07_03',
        '2017_11_24', '2017_12_24', '2018_07_03', '2018_11_23', '2018_12_24',
        '2019_07_03', '2019_11_29', '2019_12_24', '2020_11_27', '2020_12_24',
        '2021_11_26', '2021_12_24', '2022_11_25', '2022_12_24', '2023_07_03',
        '2023_11_24', '2023_12_24', '2024_07_03', '2024_11_29', '2024_12_24',
        '2025_07_03', '2025_11_28', '2025_12_24', '2026_11_27', '2026_12_24',
        '2027_11_26', '2027_12_24'
    ]
    
    # Get the day from the filename
    day = day_to_check.replace('.txt', '').replace('.parquet', '')
    
    # Determine closing time
    if day in early_closing_days:
        end_time = datetime.strptime('12:59:59.999', '%H:%M:%S.%f').time()
    else:
        end_time = datetime.strptime('15:59:59.999', '%H:%M:%S.%f').time()
    
    start_time = datetime.strptime('09:30:00.000', '%H:%M:%S.%f').time()
    
    filtered_df = df.copy()
    
    if 'time' in filtered_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['time']):
            try:
                filtered_df['time'] = pd.to_datetime(filtered_df['time'], format='mixed', errors='coerce')
            except Exception as e:
                logger.error(f"Error converting time to datetime: {e}")
                return filtered_df
        
        time_only = filtered_df['time'].dt.time
        filtered_df = filtered_df[(time_only >= start_time) & (time_only <= end_time)]
    
    return filtered_df

def add_outliers_column(df, filtered_df):
    """Add outliers column - identical logic to original"""
    prices = filtered_df['price'].values
    outliers = detect_outliers_numba(prices)

    # Convert positions to absolute indices
    outlier_indices = filtered_df.index[outliers]

    # Set is_not_outlier to NaN where needed
    filtered_df['is_not_outlier'] = np.where(filtered_df.index.isin(outlier_indices), np.nan, 1)

    # Initialize everything to 0 in the original df
    df['is_not_outlier'] = 0
    df.loc[filtered_df.index, 'is_not_outlier'] = filtered_df['is_not_outlier']

    return df

# ====================================================================
# Processing Functions
# ====================================================================

def process_single_file(s3_processor, symbol, asset_type, filename, file_format, start_date=None):
    """Process a single file - identical logic to original"""
    try:
        # Check if we should skip based on start_date
        if start_date:
            file_date = filename.replace('.txt', '').replace('.parquet', '')
            if file_date < start_date:
                return "skipped"
        
        # Read the file
        df = s3_processor.read_data_file(symbol, asset_type, filename, file_format)
        if df is None:
            return "error"
        
        # Check if file already has 5 columns (already processed)
        if len(df.columns) == 5:
            logger.debug(f"Skipping {symbol}/{filename} - already has 5 columns")
            return "skipped"
        
        # Process the data - identical logic
        data = prepare_data(df, file_format)
        filtered_df = filter_trading_hours(data, filename)
        processed_df = add_outliers_column(data, filtered_df)
        
        # Write back to S3
        success = s3_processor.write_data_file(processed_df, symbol, asset_type, filename, file_format)
        
        return "processed" if success else "error"
        
    except Exception as e:
        logger.error(f"Error processing {symbol}/{filename}: {e}")
        return "error"

def process_single_symbol(s3_processor, symbol, asset_type, file_format, max_workers=None):
    """Process all files for a single symbol"""
    try:
        logger.info(f"Processing symbol: {symbol}")
        
        # Get start date from last update info
        start_date = s3_processor.get_last_update_info(symbol, asset_type)
        if start_date:
            logger.info(f"Symbol {symbol} - Start date: {start_date}")
        
        # Get list of files to process
        files = s3_processor.list_symbol_files(symbol, asset_type, file_format)
        
        if not files:
            logger.warning(f"No files found for {symbol}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "total_files": 0}
        
        # Filter by start_date if provided
        if start_date:
            files = [f for f in files if f.replace('.txt', '').replace('.parquet', '') >= start_date]
        
        if not files:
            logger.info(f"No new files to process for {symbol}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "total_files": 0}
        
        logger.info(f"Processing {len(files)} files for {symbol}")
        
        # Process files
        processed = 0
        skipped = 0
        errors = 0
        
        for filename in files:
            result = process_single_file(s3_processor, symbol, asset_type, filename, file_format, start_date)
            if result == "processed":
                processed += 1
            elif result == "skipped":
                skipped += 1
            else:  # "error"
                errors += 1
        
        logger.info(f"Symbol {symbol} completed: {processed} processed, {skipped} skipped, {errors} errors")
        
        return {
            "symbol": symbol,
            "processed": processed, 
            "skipped": skipped, 
            "errors": errors,
            "total_files": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error processing symbol {symbol}: {e}")
        return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, "total_files": 0}

def process_all_symbols(symbols, asset_type, file_format, max_workers=None):
    """Process all symbols using ThreadPoolExecutor"""
    start_time = time.time()
    
    # Initialize S3 processor
    s3_processor = S3DataProcessor()
    
    logger.info(f"Starting outliers processing for {len(symbols)} symbols")
    logger.info(f"Asset type: {asset_type}, File format: {file_format}")
    
    results = []
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(len(symbols), os.cpu_count() or 4, 8)  # Max 8 workers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each symbol for processing
        futures = {
            executor.submit(process_single_symbol, s3_processor, symbol, asset_type, file_format, max_workers): symbol 
            for symbol in symbols
        }
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                progress = len(results) / len(symbols) * 100
                logger.info(f"Progress: {len(results)}/{len(symbols)} ({progress:.0f}%) - {result['symbol']}: {result['processed']} processed")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results.append({"symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, "total_files": 0})
    
    # Summary
    total_time = time.time() - start_time
    total_processed = sum(r["processed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_files = sum(r["total_files"] for r in results)
    
    logger.info("====== OUTLIERS PROCESSING SUMMARY ======")
    logger.info(f"Completed in: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    logger.info(f"Total files analyzed: {total_files}")
    logger.info(f"  - Processed successfully: {total_processed}")
    logger.info(f"  - Skipped (already processed): {total_skipped}")
    logger.info(f"  - Errors: {total_errors}")
    logger.info(f"Symbols: {len(results)} total")
    logger.info(f"Average speed: {total_processed / max(1, total_time):.2f} files/s")
    logger.info("==========================================")
    
    return results

# ====================================================================
# Configuration and Main Functions
# ====================================================================

def load_config():
    """Load configuration with embedded symbol lists"""
    # Resolve environment variables  
    base_dir = wmill.get_variable("u/niccolosalvini27/BASE_DIR")

    # The entire configuration as a Python dictionary
    config = {
        "general": {
            "base_dir": f"{base_dir}/data", 
            "file_format": "parquet",
            "outliers_threads_max": 8
        },
        "pipelines": {
            "stocks_batch1": {
                "enabled": True,
                "asset_type": "stocks",
                "symbols": ["GE", "JNJ"],
                "steps": ["outliers"]
            },
            "stocks_batch2": {
                "enabled": False,
                "asset_type": "stocks", 
                "symbols": [],
                "steps": ["outliers"]
            },
            "ETFs": {
                "enabled": False,
                "asset_type": "ETFs",
                "symbols": [],
                "steps": ["outliers"]
            }
        }
    }
            
    return config

def main():
    parser = argparse.ArgumentParser(description='Standalone Outliers Detection for ForVARD Project')
    parser.add_argument('--pipelines', nargs='+', help='Pipelines to run (e.g., stocks_batch1 ETFs)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    safe_print(f"OUTLIERS PROCESSING START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", "PHASE")
    
    try:
        config = load_config()
        
        if args.pipelines:
            pipelines_to_run = args.pipelines
        else:
            pipelines_to_run = [name for name, conf in config['pipelines'].items() if conf.get('enabled', True)]
        
        if not pipelines_to_run:
            safe_print("No pipeline to run. Check configuration.", "ERROR")
            sys.exit(1)
        
        safe_print(f"Pipelines to execute: {pipelines_to_run}", "INFO")
        
        results = {}
        for pipeline_name in pipelines_to_run:
            if pipeline_name not in config['pipelines'] or not config['pipelines'][pipeline_name].get('enabled', True):
                safe_print(f"Pipeline '{pipeline_name}' disabled or not found. Skipping.", "INFO")
                continue
            
            pipeline_config = config['pipelines'][pipeline_name]
            
            # Only process if outliers is in steps and asset type is stocks/ETFs
            if 'outliers' not in pipeline_config.get('steps', []):
                safe_print(f"Pipeline '{pipeline_name}' does not include outliers step. Skipping.", "INFO")
                continue
                
            if pipeline_config['asset_type'].lower() not in ['stocks', 'etfs']:
                safe_print(f"Pipeline '{pipeline_name}' asset type '{pipeline_config['asset_type']}' does not require outliers processing. Skipping.", "INFO")
                continue
            
            safe_print(f"Starting outliers processing for {pipeline_name.upper()}", "PHASE")
            safe_print(f"Symbols: {len(pipeline_config['symbols'])}, Type: {pipeline_config['asset_type']}", "INFO")
            
            if not pipeline_config['symbols']:
                safe_print(f"No symbols found for {pipeline_name}. Skipping.", "WARNING")
                results[pipeline_name] = True
                continue
            
            # Calculate optimal workers
            max_workers = min(
                config['general'].get('outliers_threads_max', 8),
                len(pipeline_config['symbols']),
                os.cpu_count() or 4
            )
            
            # Process outliers
            outliers_results = process_all_symbols(
                pipeline_config['symbols'],
                pipeline_config['asset_type'],
                config['general']['file_format'],
                max_workers
            )
            
            # Check if successful
            total_errors = sum(r["errors"] for r in outliers_results)
            pipeline_success = total_errors == 0
            results[pipeline_name] = pipeline_success
            
            if pipeline_success:
                safe_print(f"Outliers processing completed successfully for {pipeline_name}", "COMPLETE")
            else:
                safe_print(f"Outliers processing completed with {total_errors} errors for {pipeline_name}", "ERROR")
        
        end_time = datetime.now()
        safe_print("FINAL SUMMARY", "PHASE")
        safe_print(f"Total time: {end_time - start_time}")
        
        for pipeline, success in results.items():
            safe_print(f"{pipeline:>10}: {'SUCCESS' if success else 'FAILED'}")
        
        if all(results.values()):
            safe_print("\\nALL PIPELINES SUCCESSFULLY COMPLETED!", "PHASE")
            sys.exit(0)
        else:
            safe_print("\\nSOME PIPELINES FAILED!", "PHASE")
            sys.exit(1)

    except Exception as e:
        safe_print(f"CRITICAL ERROR: {e}", "ERROR")
