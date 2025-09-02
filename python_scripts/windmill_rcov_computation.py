"""
ForVARD Project - Windmill Realized Covariance Computation
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 01/07/2025

2025 University of Messina, Department of Economics.
Research code - Unauthorized distribution prohibited.

Windmill-adapted script that:
1. Reads processed tick data from S3 datalake
2. Computes realized covariance measures using IDENTICAL logic to compute_rcov.py
3. Saves results to S3 in CSV format
"""

import os
import sys
import time
import logging
import threading
import traceback
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import re
from pathlib import Path
from numba import jit, prange

# S3/MinIO imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Windmill and Slack imports
import requests
import wmill

# Thread-safe lock per l'accesso ai file
_file_lock = threading.Lock()

# --- Slack Notification Function ---
def send_slack_notification(message_text):
    """
    Sends a notification message to a configured Slack channel.
    
    Reads the Slack API token and Channel ID from environment variables
    set in Windmill.
    """
    slack_token = wmill.get_variable("u/niccolosalvini27/SLACK_API_TOKEN")
    slack_channel = wmill.get_variable("u/niccolosalvini27/SLACK_CHANNEL_ID")

    if not slack_token or not slack_channel:
        print("Variabili d'ambiente Slack non trovate (SLACK_API_TOKEN, SLACK_CHANNEL_ID). Notifica saltata.")
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

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Lancia un'eccezione per risposte HTTP 4xx/5xx
        
        response_json = response.json()
        if response_json.get("ok"):
            print("Notifica di riepilogo inviata a Slack con successo.")
        else:
            print(f"Errore nell'invio a Slack: {response_json.get('error')}")

    except requests.exceptions.RequestException as e:
        print(f"Errore di rete durante l'invio della notifica a Slack: {e}")
    except Exception as e:
        print(f"Errore imprevisto durante l'invio a Slack: {e}")

# ============================================================
# EMBEDDED RV LIBRARY FUNCTIONS (from rv_library.py)
# ============================================================

def detect_market_open(first_data_time):
    """Detect market opening time based on first data point - Fixed forex logic."""
    first_hour = first_data_time.hour
    
    if 9 <= first_hour <= 10:  # Stock market
        return first_data_time.replace(hour=9, minute=30, second=0, microsecond=0)
    elif first_hour in [0, 16, 17, 18]:  # Futures/forex - BUT check if we actually need backfill
        potential_open = first_data_time.replace(hour=first_hour, minute=0, second=0, microsecond=0)
        
        # For forex: if first data is far from hour boundary, just use first data time
        minutes_from_hour = first_data_time.minute + first_data_time.second/60 + first_data_time.microsecond/60000000
        if minutes_from_hour > 30:  # More than 30 minutes into the hour
            # Just round down to nearest 5-minute boundary
            rounded_minute = (first_data_time.minute // 5) * 5
            return first_data_time.replace(minute=rounded_minute, second=0, microsecond=0)
        else:
            return potential_open
    else:  # Unknown market - use first data time rounded down
        rounded_minute = (first_data_time.minute // 5) * 5
        return first_data_time.replace(minute=rounded_minute, second=0, microsecond=0)

def resample_prices(prices, resample_freq='5T', resampling_method='last', 
                           origin_offset_minutes=0):
    """
    Unified resampling with SMART backfill logic - no excessive backfill for forex.
    """
    
    first_data_time = prices.index[0]
    market_open = detect_market_open(first_data_time)
    
    # Calculate effective origin (market_open + offset)
    effective_origin = market_open + pd.Timedelta(minutes=origin_offset_minutes)
    
    # SMART BACKFILL DECISION: Only backfill if gap is reasonable
    gap_minutes = (first_data_time - effective_origin).total_seconds() / 60
    should_backfill = abs(gap_minutes) <= 60  # Only backfill if gap <= 1 hour
    
    if not should_backfill:
        # Use first data time as starting point instead
        rounded_minute = (first_data_time.minute // 5) * 5
        effective_origin = first_data_time.replace(minute=rounded_minute, second=0, microsecond=0)
        effective_origin = effective_origin + pd.Timedelta(minutes=origin_offset_minutes)
    
    # Perform resampling with origin
    resample_kwargs = {
        'label': 'right',
        'closed': 'right',
        'origin': effective_origin
    }
    
    if resampling_method == 'mean':
        resampled = prices.resample(resample_freq, **resample_kwargs).mean()
    elif resampling_method == 'last':
        resampled = prices.resample(resample_freq, **resample_kwargs).last()
    elif resampling_method == 'median':
        resampled = prices.resample(resample_freq, **resample_kwargs).median()
    else:
        raise ValueError(f"Method '{resampling_method}' not supported.")
    
    # Create grid only from first actual data point forward
    if should_backfill:
        start_time = effective_origin
    else:
        start_time = resampled.index[0]  # Start from first actual resampled point
    
    end_time = resampled.index[-1]
    complete_grid = pd.date_range(start=start_time, end=end_time, freq=resample_freq)
    
    # Reindex to complete grid
    resampled = resampled.reindex(complete_grid)
    
    # Count REAL observations (before any filling)
    real_observations_before_first_data = (resampled.index < first_data_time).sum()
    total_real_observations = resampled.count()
    
    # CORRECTED M calculation
    if should_backfill and real_observations_before_first_data > 0:
        # We have reasonable backfill: the first backfill point counts as real
        M = total_real_observations
    else:
        # No backfill or excessive backfill avoided: standard counting
        M = total_real_observations - 1
    
    # Apply backfill only if reasonable
    if should_backfill:
        first_available_price = prices.iloc[0]
        mask_before_data = resampled.index < first_data_time
        resampled.loc[mask_before_data] = first_available_price
    
    # Forward fill any remaining NaN
    resampled = resampled.ffill().dropna()
    
    return resampled, M

def replace_nan(df):
    """
    Replace NaN values in the price column with local averages.
    Uses a window of 2 values before and after each NaN.
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Find indices with NaN values
    nan_indices = df_copy.index[df_copy['price'].isna()]
    
    for idx in nan_indices:
        # Get position in the dataframe
        pos = df_copy.index.get_loc(idx)
        
        # Get surrounding values (2 before and 2 after)
        start_pos = max(0, pos - 2)
        end_pos = min(len(df_copy), pos + 3)
        
        surrounding_values = pd.concat([
            df_copy['price'].iloc[start_pos:pos],
            df_copy['price'].iloc[pos+1:end_pos]
        ]).dropna()
        
        # Replace NaN with mean if surrounding values exist
        if len(surrounding_values) > 0:
            df_copy.at[idx, 'price'] = surrounding_values.mean()
    
    return df_copy

def prepare_data_rv(file_path, config):
    """
    Load and prepare tick data from file.
    """
    # Determine column types and names based on asset type
    if config['asset_type'] in ['forex', 'futures']:
        data_type = {0: 'str', 1: 'float', 2: 'float', 3: 'float', 4: 'int', 5: 'int'}
        data_columns = ['time', 'price', 'bid', 'ask', 'volume', 'trades']
    else:
        data_type = {0: 'str', 1: 'float', 2: 'int', 3: 'int', 4: 'float'}
        data_columns = ['time', 'price', 'volume', 'trades', 'no_outliers']

    # Get file format from config, default to 'txt'
    file_format = config.get('file_format', 'txt').lower()
   
    # Read the file based on format
    if file_format == 'parquet':
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Ensure column names are consistent
        if len(df.columns) == len(data_columns):
            df.columns = data_columns
    else:
        # Read CSV (txt) file
        df = pd.read_csv(file_path, header=None, dtype=data_type)
        df.columns = data_columns

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
       
    # Convert time to datetime and set as index
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
    df.set_index('time', inplace=True)
    
    # Process price data based on asset type
    if config['asset_type'] in ['forex', 'futures'] and 'bid' in df.columns and 'ask' in df.columns:
        # For forex/futures with bid-ask data
        df['price'] = (df['bid'] + df['ask']) / 2
    else:
        # Adjust price based on outliers
        # if 'no_outliers' = 0 : overnight, if 'no_outliers' = NaN : outliers, if 'no_outliers' = 1 : good price
        df = df[(df['no_outliers'] != 0) | df['no_outliers'].isna()]
        df['price'] = df['price'] * df['no_outliers']
        # Replace NaN values with local averages
        df = replace_nan(df)
    
    # Remove any remaining NaN values
    df = df.dropna(subset=['price'])
    
    return df

# ============================================================
# EMBEDDED RCOV LIBRARY FUNCTIONS (from rcov_library.py)
# ============================================================

def filter_trading_hours(df, early_closing_day_file, date):
    """
    Filter DataFrame based on trading hours with caching for early closing days.
    """
    # Cache early closing days to avoid repeated file reads
    if not hasattr(filter_trading_hours, '_early_closing_cache'):
        try:
            with open(early_closing_day_file, 'r') as f:
                filter_trading_hours._early_closing_cache = set(line.strip() for line in f)
        except FileNotFoundError:
            filter_trading_hours._early_closing_cache = set()
    
    # Pre-compute time objects (avoid repeated parsing)
    if not hasattr(filter_trading_hours, '_time_cache'):
        filter_trading_hours._time_cache = {
            'start': datetime.strptime('09:30:00.000', '%H:%M:%S.%f').time(),
            'end_regular': datetime.strptime('15:59:59.999', '%H:%M:%S.%f').time(),
            'end_early': datetime.strptime('12:59:59.999', '%H:%M:%S.%f').time()
        }
    
    # Determine closing time
    end_time = (filter_trading_hours._time_cache['end_early'] 
                if date in filter_trading_hours._early_closing_cache 
                else filter_trading_hours._time_cache['end_regular'])
    
    # Vectorized time filtering
    time_mask = (df.index.time >= filter_trading_hours._time_cache['start']) & (df.index.time <= end_time)
    return df[time_mask]

def calculate_returns_from_prices(sync_prices):
    """
    Calculate log returns from synchronized prices - optimized version.
    """
    # Vectorized log return calculation for all columns at once
    log_prices = np.log(sync_prices.values)
    log_returns = np.diff(log_prices, axis=0)
    
    # Create DataFrame directly from numpy array
    return pd.DataFrame(
        log_returns, 
        index=sync_prices.index[1:], 
        columns=sync_prices.columns
    )

@jit(nopython=True, cache=True)
def _calculate_rcov_numba(returns_array):
    """
    Numba-optimized realized covariance calculation.
    """
    n_periods, n_assets = returns_array.shape
    rcov = np.zeros((n_assets, n_assets))
    
    for k in range(n_periods):
        r_k = returns_array[k, :]
        # Vectorized outer product
        for i in range(n_assets):
            for j in range(n_assets):
                rcov[i, j] += r_k[i] * r_k[j]
    
    return rcov

@jit(nopython=True, cache=True)
def _calculate_bipower_variation_numba(returns_array):
    """
    Numba-optimized bipower variation calculation.
    """
    m = len(returns_array)
    if m < 2:
        return np.nan
    
    abs_returns = np.abs(returns_array)
    return_products = abs_returns[1:] * abs_returns[:-1]
    return (np.pi / 2) * np.sum(return_products)

def calculate_bipower_variation(returns):
    """
    Calculate realized bipower variation for a single asset - optimized.
    """
    if len(returns) < 2:
        return np.nan
    
    return _calculate_bipower_variation_numba(returns.values)

def calculate_bipower_covariation(returns1, returns2):
    """
    Calculate realized bipower covariation - optimized version.
    """
    # Align returns more efficiently
    mask = ~(np.isnan(returns1) | np.isnan(returns2))
    if np.sum(mask) < 2:
        return np.nan
    
    r1 = returns1[mask].values
    r2 = returns2[mask].values
    
    # Vectorized sum and difference
    r_sum = r1 + r2
    r_diff = r1 - r2
    
    # Use numba-optimized function
    rbpv_sum = _calculate_bipower_variation_numba(r_sum)
    rbpv_diff = _calculate_bipower_variation_numba(r_diff)
    
    if np.isnan(rbpv_sum) or np.isnan(rbpv_diff):
        return np.nan
    
    return 0.25 * (rbpv_sum - rbpv_diff)

@jit(nopython=True, cache=True)
def _calculate_semicovariances_numba(returns_matrix):
    """
    Numba-optimized semicovariance calculation with parallelization.
    """
    n_periods, n_assets = returns_matrix.shape
    
    # Initialize matrices
    P = np.zeros((n_assets, n_assets))
    N = np.zeros((n_assets, n_assets))
    RSCov_Mp = np.zeros((n_assets, n_assets))
    RSCov_Mn = np.zeros((n_assets, n_assets))
    
    # Parallel loop over time periods
    for k in range(n_periods):
        r_k = returns_matrix[k, :]
        
        # Positive and negative parts
        r_pos = np.maximum(r_k, 0.0)
        r_neg = np.minimum(r_k, 0.0)
        
        # Update matrices using outer products
        for i in range(n_assets):
            for j in range(n_assets):
                P[i, j] += r_pos[i] * r_pos[j]
                N[i, j] += r_neg[i] * r_neg[j]
                RSCov_Mp[i, j] += r_pos[i] * r_neg[j]
                RSCov_Mn[i, j] += r_neg[i] * r_pos[j]
    
    return P, N, RSCov_Mp, RSCov_Mn

def calculate_semicovariances(returns_matrix):
    """
    Calculate semicovariance matrices - optimized version.
    """
    if len(returns_matrix) < 1:
        return None
    
    # Use numba-optimized calculation
    P, N, RSCov_Mp, RSCov_Mn = _calculate_semicovariances_numba(returns_matrix.values)
    
    # Convert back to DataFrames
    index_cols = returns_matrix.columns
    return {
        'RSCov_P': pd.DataFrame(P, index=index_cols, columns=index_cols),
        'RSCov_N': pd.DataFrame(N, index=index_cols, columns=index_cols),
        'RSCov_Mp': pd.DataFrame(RSCov_Mp, index=index_cols, columns=index_cols),
        'RSCov_Mn': pd.DataFrame(RSCov_Mn, index=index_cols, columns=index_cols)
    }

@jit(nopython=True, cache=True, parallel=True)
def _calculate_rbpcov_matrix_numba(returns_array):
    """
    Numba-optimized bipower covariation matrix calculation.
    """
    n_assets = returns_array.shape[1]
    rbpcov = np.zeros((n_assets, n_assets))
    
    # Calculate upper triangular matrix in parallel
    for i in prange(n_assets):
        for j in range(i, n_assets):
            if i == j:
                # Diagonal: bipower variation
                rbpcov[i, j] = _calculate_bipower_variation_numba(returns_array[:, i])
            else:
                # Off-diagonal: bipower covariation
                r1 = returns_array[:, i]
                r2 = returns_array[:, j]
                
                # Handle NaN values
                mask = ~(np.isnan(r1) | np.isnan(r2))
                if np.sum(mask) < 2:
                    rbpcov[i, j] = np.nan
                else:
                    r1_clean = r1[mask]
                    r2_clean = r2[mask]
                    
                    r_sum = r1_clean + r2_clean
                    r_diff = r1_clean - r2_clean
                    
                    rbpv_sum = _calculate_bipower_variation_numba(r_sum)
                    rbpv_diff = _calculate_bipower_variation_numba(r_diff)
                    
                    if np.isnan(rbpv_sum) or np.isnan(rbpv_diff):
                        rbpcov[i, j] = np.nan
                    else:
                        rbpcov[i, j] = 0.25 * (rbpv_sum - rbpv_diff)
    
    # Fill lower triangular
    for i in range(n_assets):
        for j in range(i):
            rbpcov[i, j] = rbpcov[j, i]
    
    return rbpcov

def calculate_covariance_matrices(returns, measures=['RCov']):
    """
    Calculate all requested covariance matrices
    """
    results = {}
    returns_array = returns.values
    
    # Calculate Realized Covariance (RCov) 
    if 'RCov' in measures:
        rcov_values = _calculate_rcov_numba(returns_array)
        results['RCov'] = pd.DataFrame(
            rcov_values,
            index=returns.columns, 
            columns=returns.columns
        )
    
    # Calculate Realized Bipower Covariation (RBPCov)
    if 'RBPCov' in measures:
        rbpcov_values = _calculate_rbpcov_matrix_numba(returns_array)
        results['RBPCov'] = pd.DataFrame(
            rbpcov_values,
            index=returns.columns, 
            columns=returns.columns
        )
    
    # Calculate Realized Semicovariances (RSCov)
    if 'RSCov' in measures:
        semicov_matrices = calculate_semicovariances(returns)
        if semicov_matrices:
            results.update(semicov_matrices)
    
    return results

def process_single_day(file_paths, assets, date, asset_configs, resample_freq='1T', 
                      resampling_method='last', measures=['RCov'], logger=None,
                      mixed_asset_mode=False, early_closing_day_file=None):
    """Process a single day's data for multiple assets."""
    price_series = {}
    has_stocks = any(config['asset_type'] == 'stocks' for config in asset_configs.values())
    
    for asset in assets:
        if asset not in file_paths:
            continue
            
        try:
            config = asset_configs[asset]
            df = prepare_data_rv(file_paths[asset], config)
            
            if len(df) < 10:
                continue
            
            # Filter non-stock assets to stock trading hours when mixed mode
            if (mixed_asset_mode and has_stocks and 
                config['asset_type'] in ['forex', 'futures'] and
                early_closing_day_file and os.path.exists(early_closing_day_file)):
                
                df = filter_trading_hours(df, early_closing_day_file, date)
                if len(df) < 10:
                    continue
            
            resampled_prices, _ = resample_prices(df['price'], resample_freq, resampling_method)
            price_series[asset] = resampled_prices
            
        except Exception as e:
            if logger:
                logger.error(f"Error processing {asset} on {date}: {e}")
            continue
    
    if len(price_series) < 2:
        return None
    
    try:
        # MANTIENE l'ordine dal file di configurazione originale
        original_asset_order = list(asset_configs.keys())  # Ordine dal file symbols
        available_assets = [asset for asset in original_asset_order if asset in price_series]
        ordered_price_series = {asset: price_series[asset] for asset in available_assets}
        sync_prices = pd.DataFrame(ordered_price_series).ffill()
        
        returns = calculate_returns_from_prices(sync_prices)
        
        if len(returns) < 2:
            return None
            
        return calculate_covariance_matrices(returns, measures)
        
    except Exception as e:
        if logger:
            logger.error(f"Error calculating covariance for {date}: {e}")
        return None

def format_covariance_output(cov_matrices, date):
    """Format covariance matrices output in wide format."""
    rows = []
    first_matrix = list(cov_matrices.values())[0]
    
    for i, asset1 in enumerate(first_matrix.index):
        for j, asset2 in enumerate(first_matrix.columns):
            if i <= j:  # Upper triangular including diagonal
                row = {'date': date, 'asset1': asset1, 'asset2': asset2}
                
                for measure_name, matrix in cov_matrices.items():
                    row[measure_name] = matrix.loc[asset1, asset2]
                
                rows.append(row)
    
    return pd.DataFrame(rows)

# ============================================================
# S3 DATA PROCESSOR FOR RCOV
# ============================================================

class S3DataProcessor:
    """Handles S3 data reading and writing for RCOV computation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('rcov_processor')
        
        # Initialize S3 client
        self.s3_client = self._setup_s3_client()
        self.s3_bucket = config['s3_bucket']
        
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

    def read_data_file(self, symbol, asset_type, filename):
        """Read a data file from S3"""
        import io
        
        try:
            # Construct S3 key
            s3_key = f"data/{asset_type}/{symbol}/{filename}"
            
            # Get object from S3
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read()
            
            # Read based on file format
            file_format = self.config.get('file_format', 'txt').lower()
            
            if file_format == 'parquet':
                try:
                    import pyarrow.parquet as pq
                    df = pq.read_table(io.BytesIO(content)).to_pandas()
                    return df
                except ImportError:
                    self.logger.error("PyArrow not available for parquet reading")
                    return None
            else:
                # Read as CSV
                df = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None)
                return df
                
        except Exception as e:
            self.logger.error(f"Error reading {symbol}/{filename}: {e}")
            return None

    def save_results_to_s3(self, results_df: pd.DataFrame, output_filename: str) -> bool:
        """Save results DataFrame to S3 as CSV"""
        try:
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Upload to S3
            s3_key = f"results/{output_filename}"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_content.encode('utf-8'),
                ContentType='text/csv'
            )
            
            self.logger.info(f"Results saved to S3: {s3_key} ({len(results_df)} records)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results to S3: {e}")
            return False

    def check_existing_results(self, output_filename: str) -> Optional[pd.DataFrame]:
        """Check if results file exists in S3 and return it"""
        try:
            s3_key = f"results/{output_filename}"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            
            # Read CSV content
            import io
            df = pd.read_csv(io.StringIO(content))
            self.logger.info(f"Found existing results: {len(df)} records")
            return df
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                self.logger.info("No existing results found")
                return None
            else:
                self.logger.error(f"Error checking existing results: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Error reading existing results: {e}")
            return None

# ============================================================
# RCOV CALCULATOR CLASS (ADAPTED FROM ORIGINAL)
# ============================================================

class RCovCalculator:
    """Realized Covariance Calculator for Windmill."""
    
    def __init__(self, s3_processor: S3DataProcessor):
        self.s3_processor = s3_processor
        self.logger = logging.getLogger('rcov_processor')
        
        self.date_pattern = re.compile(r'(\d{4})_(\d{2})_(\d{2})')
        self.exclude_pattern = re.compile(r'(_last_update|adjustment)', re.IGNORECASE)

    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract YYYY_MM_DD date from filename using regex"""
        
        # Quick check to exclude files (compiled regex: _last_update|adjustment)
        if self.exclude_pattern.search(filename):
            return None
        
        # Search for date pattern (compiled regex: (\d{4})_(\d{2})_(\d{2}))
        match = self.date_pattern.search(Path(filename).stem)  # .stem delete extension
        if not match:
            return None
        
        year, month, day = match.groups()
        
        # Date validation
        try:
            y, m, d = int(year), int(month), int(day)
            if not (2000 <= y <= 2030 and 1 <= m <= 12 and 1 <= d <= 31):
                return None
            return f"{year}_{month}_{day}"
        except ValueError:
            return None

    def find_available_dates(self, asset_configs: Dict[str, Dict], 
                           start_date: str = None, end_date: str = None,
                           min_assets_threshold: float = 0.8) -> List[str]:
        """Find dates with sufficient asset coverage."""
        date_asset_counts = {}  # Dictionary to count assets per date
        total_assets = len(asset_configs)  # Total number of assets to track
        
        # Count how many assets have data for each date
        for symbol, config in asset_configs.items():
            files = self.s3_processor.list_symbol_files(symbol, config['asset_type'], config.get('file_format', 'parquet'))
            for filename in files:
                date_str = self.extract_date_from_filename(filename)  # Extract date from filename
                if date_str:
                    if date_str not in date_asset_counts:
                        date_asset_counts[date_str] = 0
                    date_asset_counts[date_str] += 1
        
        # Filter dates that have data for at least min_assets_threshold of assets
        min_assets_required = max(2, int(total_assets * min_assets_threshold))
        
        valid_dates = [
            date for date, count in date_asset_counts.items()
            if count >= min_assets_required
        ]
        
        valid_dates = sorted(valid_dates)
        
        # Apply date range filters
        if start_date:
            valid_dates = [d for d in valid_dates if d >= start_date]
        if end_date:
            valid_dates = [d for d in valid_dates if d <= end_date]
        
        self.logger.info(f"Found {len(valid_dates)} dates with >={min_assets_required}/{total_assets} assets")
        if valid_dates:
            self.logger.info(f"Date range: {valid_dates[0]} to {valid_dates[-1]}")
        
        return valid_dates

    def get_files_for_date(self, date: str, asset_configs: Dict[str, Dict]) -> Dict[str, str]:
        """Get file paths for all assets for a specific date - adapted for S3"""
        file_paths = {}
        
        for symbol, config in asset_configs.items():
            files = self.s3_processor.list_symbol_files(symbol, config['asset_type'], config.get('file_format', 'parquet'))
            
            for filename in files:
                if self.extract_date_from_filename(filename) == date:
                    # For S3, we store the filename and will read it when needed
                    file_paths[symbol] = filename
                    break
        
        return file_paths

    def process_date_parallel(self, args):
        """Process single date (for parallel execution)."""
        date, asset_configs, measures, resample_freq, resampling_method, mixed_mode = args
        
        # Get file paths for all assets on this specific date
        file_paths = self.get_files_for_date(date, asset_configs)
        if len(file_paths) < 2:  # Need at least 2 assets for covariance calculation
            print(f"[{date}] SKIP: {len(file_paths)} assets (need ≥2)")
            return None  # Skip this date if insufficient data
        
        # Convert filenames to actual file data by reading from S3
        actual_file_paths = {}
        for symbol, filename in file_paths.items():
            try:
                # Read data from S3 and save to temp file
                raw_data = self.s3_processor.read_data_file(symbol, asset_configs[symbol]['asset_type'], filename)
                if raw_data is None or raw_data.empty:
                    continue
                
                # Save to temporary file for prepare_data_rv function
                file_format = asset_configs[symbol].get('file_format', 'txt').lower()
                suffix = '.parquet' if file_format == 'parquet' else '.txt'
                
                with tempfile.NamedTemporaryFile(mode='w+b', suffix=suffix, delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    
                    if file_format == 'parquet':
                        raw_data.to_parquet(temp_file_path, index=False)
                    else:
                        raw_data.to_csv(temp_file_path, index=False, header=False)
                    
                    actual_file_paths[symbol] = temp_file_path
                    
            except Exception as e:
                print(f"[{date}] Error reading {symbol}: {e}")
                continue
        
        if len(actual_file_paths) < 2:
            # Clean up temp files
            for temp_path in actual_file_paths.values():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return None
        
        try:
            # Call external processing function
            cov_matrices = process_single_day(
                file_paths=actual_file_paths,
                assets=list(asset_configs.keys()),
                date=date,
                asset_configs=asset_configs,
                resample_freq=resample_freq,
                resampling_method=resampling_method,
                measures=measures,
                logger=None,  # No logger in parallel
                mixed_asset_mode=mixed_mode,
                early_closing_day_file=None  # Not used in S3 version
            )
            
            # Format and return results
            if cov_matrices:
                return format_covariance_output(cov_matrices, date)
            else:
                print(f"[{date}] FAIL: processing returned empty")
                return None
                
        finally:
            # Clean up temp files
            for temp_path in actual_file_paths.values():
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def run_pipeline(self, asset_configs: Dict[str, Dict], pipeline_config: Dict[str, Any], 
                    start_date: str = None, end_date: str = None,
                    force_recalc: bool = False) -> Optional[pd.DataFrame]:
        """Run RCOV pipeline - IDENTICAL STRUCTURE TO ORIGINAL"""
        
        self.logger.info(f"Running RCOV pipeline")
        start_time = time.time()  # Start performance timer
        
        if not asset_configs:
            return None
        
        # Convert dates from MM/DD/YYYY to YYYY_MM_DD internal format if provided
        final_start = start_date
        final_end = end_date
        
        if final_start:
            try:
                final_start = datetime.strptime(final_start, '%m/%d/%Y').strftime('%Y_%m_%d')
            except ValueError:
                # Already in YYYY_MM_DD format
                pass
        if final_end:
            try:
                final_end = datetime.strptime(final_end, '%m/%d/%Y').strftime('%Y_%m_%d')
            except ValueError:
                # Already in YYYY_MM_DD format
                pass
        
        # Find dates to process (only dates with sufficient asset coverage)
        dates = self.find_available_dates(asset_configs, final_start, final_end, min_assets_threshold=1.0)
        if not dates:
            self.logger.warning("No dates found")
            return None
        
        # Check existing results
        output_filename = f"rcov_{pipeline_config['pipeline_name']}.csv"
        
        existing_results = None
        if not force_recalc:
            existing_results = self.s3_processor.check_existing_results(output_filename)
            if existing_results is not None and not existing_results.empty:
                # Check which dates are missing
                existing_dates = set(existing_results['date'].unique())
                missing_dates = [d for d in dates if d not in existing_dates]
                
                if not missing_dates:
                    self.logger.info("All dates already processed")
                    return existing_results
                else:
                    dates = missing_dates
                    self.logger.info(f"Processing {len(missing_dates)} missing dates")
        
        # Extract configuration
        measures = pipeline_config.get('measures', ['RCov'])
        resample_freq = pipeline_config.get('resample_freq', '1T')
        resampling_method = pipeline_config.get('resampling_method', 'last')
        mixed_mode = pipeline_config.get('mixed_asset_mode', False)
        
        # Process dates with parallel execution
        max_workers = min(pipeline_config.get('max_workers', 4), len(dates))
        use_parallel = max_workers > 1 and len(dates) > 10
        
        self.logger.info(f"Processing {len(dates)} dates with {len(asset_configs)} assets")
        if mixed_mode:
            self.logger.info("Mixed assets: applying trading hours filter")
        if use_parallel:
            self.logger.info(f"Using {max_workers} parallel workers")
        
        results_count = 0
        all_results = []
        
        if use_parallel:
            # Parallel processing
            process_args = [
                (date, asset_configs, measures, resample_freq, resampling_method, mixed_mode) 
                for date in dates
            ]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_date = {
                    executor.submit(self.process_date_parallel, args): args[0]
                    for args in process_args
                }
                
                completed = 0
                
                for future in as_completed(future_to_date):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                        results_count += len(result)
                    
                    completed += 1
                    
                    if completed % 50 == 0 or completed == len(dates):
                        self.logger.info(f"Progress: {completed}/{len(dates)} processed, {results_count} records generated")
        else:
            # Sequential processing
            for i, date in enumerate(dates, 1):
                file_paths = self.get_files_for_date(date, asset_configs)
                if len(file_paths) < 2:  # Need at least 2 assets for covariance
                    continue
                
                # Convert filenames to actual file data by reading from S3
                actual_file_paths = {}
                temp_files = []
                
                try:
                    for symbol, filename in file_paths.items():
                        raw_data = self.s3_processor.read_data_file(symbol, asset_configs[symbol]['asset_type'], filename)
                        if raw_data is None or raw_data.empty:
                            continue
                        
                        # Save to temporary file for prepare_data_rv function
                        file_format = asset_configs[symbol].get('file_format', 'txt').lower()
                        suffix = '.parquet' if file_format == 'parquet' else '.txt'
                        
                        with tempfile.NamedTemporaryFile(mode='w+b', suffix=suffix, delete=False) as temp_file:
                            temp_file_path = temp_file.name
                            temp_files.append(temp_file_path)
                            
                            if file_format == 'parquet':
                                raw_data.to_parquet(temp_file_path, index=False)
                            else:
                                raw_data.to_csv(temp_file_path, index=False, header=False)
                            
                            actual_file_paths[symbol] = temp_file_path
                    
                    if len(actual_file_paths) < 2:
                        continue
                    
                    cov_matrices = process_single_day(
                        file_paths=actual_file_paths,
                        assets=list(asset_configs.keys()),
                        date=date,
                        asset_configs=asset_configs,
                        resample_freq=resample_freq,
                        resampling_method=resampling_method,
                        measures=measures,
                        logger=self.logger,
                        mixed_asset_mode=mixed_mode,
                        early_closing_day_file=None  # Not used in S3 version
                    )
                    
                    if cov_matrices:
                        result = format_covariance_output(cov_matrices, date)
                        all_results.append(result)
                        results_count += len(result)
                
                finally:
                    # Clean up temp files
                    for temp_path in temp_files:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                
                if i % 10 == 0 or i == len(dates):
                    self.logger.info(f"Progress: {i}/{len(dates)} processed, {results_count} records generated")
        
        # Combine results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            
            # If we had existing results, combine them
            if existing_results is not None and not existing_results.empty:
                final_df = pd.concat([existing_results, final_df], ignore_index=True)
            
            # Sort results by date and asset pairs
            final_df = final_df.sort_values(['date', 'asset1', 'asset2'], ignore_index=True)
            
            # Save to S3
            if self.s3_processor.save_results_to_s3(final_df, output_filename):
                success_rate = results_count / len(dates) * 100 if dates else 0
                self.logger.info(f"SUCCESS: {len(final_df)} total records saved")
                self.logger.info(f"Success rate: {success_rate:.1f}%")
                self.logger.info(f"Total execution time: {time.time() - start_time:.1f} seconds")
                return final_df
            else:
                self.logger.error("Failed to save results to S3")
                return None
        else:
            self.logger.warning("No successful results")
            return None

# ============================================================
# CORE PROCESSING FUNCTIONS
# ============================================================

def setup_logging() -> logging.Logger:
    """Setup logging configuration with console output only"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger('rcov_processor')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger

def parse_symbols_list(symbols: List[str], asset_type: str, file_format: str) -> Dict[str, Dict]:
    """Parse symbols list into asset configs format"""
    asset_configs = {}
    
    for symbol in symbols:
        symbol = symbol.upper().strip()
        if symbol:
            asset_configs[symbol] = {
                'asset_type': asset_type,
                'file_format': file_format
            }
    
    return asset_configs

# ============================================================
# MAIN FUNCTION FOR WINDMILL
# ============================================================

def main(
    # Pipeline configuration
    pipeline_name="realized_covariance",
    pipeline_enabled=True,
    
    # Mixed assets configuration
    mixed_enabled=False,
    mixed_symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V", 
                   "EURUSD", "GBPUSD", "AUDUSD", "JPYUSD", "CHFUSD",
                   "ES", "CL", "GC", "NG", "NQ"],
    
    # Individual asset type configuration
    stocks_enabled=True,
    stocks_symbols=[ "MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY", "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA", "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM", "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK", "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC" ]
    forex_enabled=True,
    forex_symbols=["EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "JPYUSD", "CHFUSD", "SGDUSD", "HKDUSD", "KRWUSD", "INRUSD", "RUBUSD", "BRLUSD"],
    futures_enabled=True,
    futures_symbols=["ES", "CL", "GC", "NG", "NQ", "TY", "FV", "EU", "SI", "C", "W", "VX"],
    etfs_enabled=True,
    etfs_symbols=["AGG", "BND", "GLD", "SLV", "SUSA", "EFIV", "ESGV", "ESGU", "AFTY", "MCHI", "EWH","EEM", "IEUR", "VGK", "FLCH", "EWJ", "NKY", "EWZ", "EWC", "EWU", "EWI", "EWP","ACWI", "IOO", "GWL", "VEU", "IJH", "MDY", "IVOO", "IYT", "XTN", "XLI", "XLU", "VPU", "SPSM", "IJR", "VIOO", "QQQ", "ICLN", "ARKK", "SPLG", "SPY", "VOO", "IYY", "VTI", "DIA"],
    
    # Processing configuration
    measures=["RCov", "RBPCov"],
    resample_freq="1T",
    resampling_method="last",
    max_workers=4,
    file_format="parquet",
    
    # Date range (optional)
    start_date=None,  # Format: "MM/DD/YYYY" or "YYYY_MM_DD"
    end_date=None,    # Format: "MM/DD/YYYY" or "YYYY_MM_DD"
    force_recalc=False,
    
    s3_bucket=None
):
    """
    Windmill Realized Covariance Computation
    
    Reads processed tick data from S3, computes realized covariance measures using
    IDENTICAL logic to compute_rcov.py, and saves results to S3 as CSV.
    
    Args:
        pipeline_name: Name of the pipeline
        pipeline_enabled: Whether the pipeline is enabled
        
        # Mixed assets (recommended for covariance)
        mixed_enabled: Enable mixed assets pipeline (stocks + forex + futures)
        mixed_symbols: List of mixed symbols for covariance calculation
        
        # Individual asset types
        stocks_enabled: Enable stocks-only pipeline
        stocks_symbols: List of stock symbols
        forex_enabled: Enable forex-only pipeline  
        forex_symbols: List of forex symbols
        futures_enabled: Enable futures-only pipeline
        futures_symbols: List of futures symbols
        etfs_enabled: Enable ETFs-only pipeline
        etfs_symbols: List of ETF symbols
        
        # Processing configuration
        measures: List of covariance measures to calculate
        resample_freq: Resampling frequency (e.g., '1T', '5T')
        resampling_method: Resampling method ('last', 'mean', 'median')
        max_workers: Maximum number of parallel workers
        file_format: File format ('parquet' or 'txt')
        
        # Date range
        start_date: Start date (MM/DD/YYYY or YYYY_MM_DD)
        end_date: End date (MM/DD/YYYY or YYYY_MM_DD)
        force_recalc: Force recalculation of existing results
        
        s3_bucket: S3 bucket name (if None, will use Windmill variable)
    """
    start_time = datetime.now()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"REALIZED COVARIANCE COMPUTATION START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not pipeline_enabled:
        logger.info("Pipeline is disabled. Exiting.")
        return
    
    try:
        # Get S3 bucket from Windmill if not provided
        if s3_bucket is None:
            s3_bucket = wmill.get_variable("u/niccolosalvini27/S3_BUCKET")
        
        # Initialize S3 processor
        s3_config = {
            's3_bucket': s3_bucket,
            'file_format': file_format
        }
        s3_processor = S3DataProcessor(s3_config)
        
        # Initialize calculator
        calculator = RCovCalculator(s3_processor)
        
        # Process all enabled pipelines
        all_results = []
        total_records = 0
        total_errors = 0
        
        # Process mixed assets if enabled (RECOMMENDED for covariance)
        if mixed_enabled and mixed_symbols:
            logger.info("Processing MIXED ASSETS pipeline...")
            
            # Parse mixed symbols (assume they are mixed stocks/forex/futures)
            # For simplicity, we'll auto-detect asset types based on symbol patterns
            asset_configs = {}
            for symbol in mixed_symbols:
                symbol = symbol.upper().strip()
                if not symbol:
                    continue
                
                # Simple heuristics for asset type detection
                if len(symbol) == 6 and 'USD' in symbol:
                    asset_type = 'forex'
                elif len(symbol) <= 2 or symbol in ['ES', 'CL', 'GC', 'NG', 'NQ', 'TY', 'FV', 'EU', 'SI', 'C', 'W', 'VX']:
                    asset_type = 'futures'
                elif len(symbol) == 3 and symbol.isupper():
                    asset_type = 'etfs'
                else:
                    asset_type = 'stocks'
                
                asset_configs[symbol] = {
                    'asset_type': asset_type,
                    'file_format': file_format
                }
            
            pipeline_config = {
                'pipeline_name': f"{pipeline_name}_mixed",
                'measures': measures,
                'resample_freq': resample_freq,
                'resampling_method': resampling_method,
                'max_workers': max_workers,
                'mixed_asset_mode': True
            }
            
            try:
                mixed_results = calculator.run_pipeline(
                    asset_configs, pipeline_config, start_date, end_date, force_recalc
                )
                if mixed_results is not None:
                    all_results.append(('mixed', mixed_results))
                    total_records += len(mixed_results)
                    logger.info(f"Mixed assets completed: {len(mixed_results)} records")
                else:
                    total_errors += 1
                    logger.error("Mixed assets pipeline failed")
            except Exception as e:
                logger.error(f"Error in mixed assets pipeline: {e}")
                total_errors += 1
        
        # Process individual asset types if enabled
        asset_type_configs = [
            ('stocks', stocks_enabled, stocks_symbols),
            ('forex', forex_enabled, forex_symbols),
            ('futures', futures_enabled, futures_symbols),
            ('etfs', etfs_enabled, etfs_symbols)
        ]
        
        for asset_type, enabled, symbols in asset_type_configs:
            if enabled and symbols:
                logger.info(f"Processing {asset_type.upper()} pipeline...")
                
                asset_configs = parse_symbols_list(symbols, asset_type, file_format)
                
                pipeline_config = {
                    'pipeline_name': f"{pipeline_name}_{asset_type}",
                    'measures': measures,
                    'resample_freq': resample_freq,
                    'resampling_method': resampling_method,
                    'max_workers': max_workers,
                    'mixed_asset_mode': False
                }
                
                try:
                    results = calculator.run_pipeline(
                        asset_configs, pipeline_config, start_date, end_date, force_recalc
                    )
                    if results is not None:
                        all_results.append((asset_type, results))
                        total_records += len(results)
                        logger.info(f"{asset_type.capitalize()} completed: {len(results)} records")
                    else:
                        total_errors += 1
                        logger.error(f"{asset_type.capitalize()} pipeline failed")
                except Exception as e:
                    logger.error(f"Error in {asset_type} pipeline: {e}")
                    total_errors += 1
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("FINAL SUMMARY")
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Pipeline: {pipeline_name}")
        logger.info(f"Total records: {total_records}, Errors: {total_errors}")
        
        # Determine status
        if total_errors == 0 and total_records > 0:
            logger.info("ALL PROCESSING COMPLETED SUCCESSFULLY!")
            status_emoji = "✅"
            status_text = "SUCCESS"
        elif total_records > 0:
            logger.warning("SOME PROCESSING ERRORS OCCURRED!")
            status_emoji = "⚠️"
            status_text = "PARTIAL SUCCESS"
        else:
            logger.error("NO SUCCESSFUL PROCESSING!")
            status_emoji = "❌"
            status_text = "FAILURE"
        
        # Create Slack message
        slack_message = f"{status_emoji} *REALIZED COVARIANCE COMPUTATION COMPLETED* {status_emoji}\n\n"
        slack_message += f"*Status:* {status_text}\n"
        slack_message += f"*Pipeline:* {pipeline_name}\n"
        slack_message += f"*Duration:* {str(duration).split('.')[0]}\n"
        slack_message += f"*Pipelines processed:* {len(all_results)}\n\n"
        
        slack_message += "*Processing Summary:*"
        slack_message += f"\n📊 Total covariance records: {total_records}"
        slack_message += f"\n❌ Pipeline errors: {total_errors}"
        
        # Add details per pipeline
        if all_results:
            slack_message += f"\n\n*Pipeline Details:*"
            for pipeline_type, results in all_results:
                slack_message += f"\n• {pipeline_type.capitalize()}: {len(results)} records"
        
        slack_message += f"\n\n*Measures calculated:* {', '.join(measures)}"
        slack_message += f"\n*Resampling:* {resample_freq} ({resampling_method})"
        slack_message += f"\n*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send Slack notification
        try:
            send_slack_notification(slack_message)
        except Exception as e:
            logger.warning(f"Error sending Slack notification: {e}")
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
        traceback.print_exc()

