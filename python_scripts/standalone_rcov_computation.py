#!/usr/bin/env python3
"""
Standalone Realized Covariance Computation Script
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

This script computes realized covariance measures for asset pairs using data from S3 datalake
and saves results to PostgreSQL database. It includes identical logic from the main pipeline
but operates as a standalone script.

Author: AI Assistant
Date: 2025-01-19
"""

import os
import sys
import json
import time
import boto3
import logging
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from numba import jit
import wmill
import pyarrow
import requests
WINDMILL_AVAILABLE = True


# Slack notification function
def send_slack_notification(message_text):
    """
    Sends a notification message to a configured Slack channel.
    
    Reads the Slack API token and Channel ID from environment variables
    set in Windmill.
    """
    slack_token = wmill.get_variable("u/niccolosalvini27/SLACK_API_TOKEN")
    slack_channel = wmill.get_variable("u/niccolosalvini27/SLACK_CHANNEL_ID")

    if not slack_token or not slack_channel:
        logger.warning("Slack environment variables not found (SLACK_API_TOKEN, SLACK_CHANNEL_ID). Notification skipped.")
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
        import requests
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        response_json = response.json()
        if response_json.get("ok"):
            logger.info("Slack notification sent successfully.")
        else:
            logger.error(f"Slack API error: {response_json.get('error')}")

    except ImportError:
        logger.error("Error sending Slack notification: No module named 'requests'")
    except Exception as e:
        logger.error(f"Error sending Slack notification: {e}")

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('standalone_rcov')

# ============================================================================
# REALIZED COVARIANCE COMPUTATION LIBRARY (Embedded from rcov_library.py)
# ============================================================================

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
def _calculate_rbpcov_matrix_numba(returns_array):
    """
    Numba-optimized realized bipower covariation calculation for full matrix.
    """
    n_periods, n_assets = returns_array.shape
    rbpcov = np.zeros((n_assets, n_assets))
    
    for k in range(1, n_periods):
        r_k = returns_array[k, :]
        r_k_minus_1 = returns_array[k-1, :]
        
        for i in range(n_assets):
            for j in range(n_assets):
                rbpcov[i, j] += np.abs(r_k[i]) * np.abs(r_k_minus_1[j])
    
    # Scaling factor for bipower variation
    mu1 = np.sqrt(2.0 / np.pi)
    scaling_factor = np.pi / 2.0 / (mu1 ** 2)
    
    return rbpcov * scaling_factor

@jit(nopython=True, cache=True)
def _calculate_semicovariances_numba(returns_array):
    """
    Numba-optimized calculation of semicovariance matrices.
    """
    n_periods, n_assets = returns_array.shape
    P = np.zeros((n_assets, n_assets))
    N = np.zeros((n_assets, n_assets))
    RSCov_Mp = np.zeros((n_assets, n_assets))
    RSCov_Mn = np.zeros((n_assets, n_assets))
    
    for k in range(n_periods):
        r_k = returns_array[k, :]
        
        for i in range(n_assets):
            for j in range(n_assets):
                r_i = r_k[i]
                r_j = r_k[j]
                
                # Positive semicovariance: both returns positive
                if r_i >= 0 and r_j >= 0:
                    P[i, j] += r_i * r_j
                
                # Negative semicovariance: both returns negative
                if r_i <= 0 and r_j <= 0:
                    N[i, j] += r_i * r_j
                
                # Mixed positive: i positive, j negative
                if r_i >= 0 and r_j <= 0:
                    RSCov_Mp[i, j] += r_i * r_j
                
                # Mixed negative: i negative, j positive
                if r_i <= 0 and r_j >= 0:
                    RSCov_Mn[i, j] += r_i * r_j
    
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

def resample_prices(prices, resample_freq='1T', resampling_method='last', 
                           origin_offset_minutes=0):
    """
    Unified resampling with SMART backfill logic - no excessive backfill for forex.
    This function is identical to the original from rv_library.py
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
    
    # Forward fill missing values
    resampled = resampled.ffill()
    
    # Drop NaN values
    resampled = resampled.dropna()
    
    return resampled, M

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

def prepare_data_rcov(file_path, config, s3_client, bucket_name, logger=None):
    """
    Load and prepare tick data from S3 file.
    Enhanced version with robust column mapping similar to standalone_rv_computation.py
    """
    # Determine column types and names based on asset type
    if config['asset_type'] in ['forex', 'futures']:
        data_type = {0: 'str', 1: 'float', 2: 'float', 3: 'float', 4: 'int', 5: 'int'}
        data_columns = ['time', 'price', 'bid', 'ask', 'volume', 'trades']
    else:
        # Use is_not_outlier instead of no_outliers to match original libraries
        data_type = {0: 'str', 1: 'float', 2: 'int', 3: 'int', 4: 'float'}
        data_columns = ['time', 'price', 'volume', 'trades', 'is_not_outlier']

    # Get file format from config, default to 'parquet'
    file_format = config.get('file_format', 'parquet').lower()
    
    # Use logger if available, otherwise print
    def log_message(level, message):
        if logger:
            if level == 'info':
                logger.info(f"prepare_data_rcov: {message}")
            elif level == 'warning':
                logger.warning(f"prepare_data_rcov: {message}")
            elif level == 'error':
                logger.error(f"prepare_data_rcov: {message}")
        else:
            print(f"prepare_data_rcov: {message}")
   
    try:
        log_message('info', f"Loading {file_path} with format {file_format}, asset_type {config['asset_type']}")
        
        # Read the file based on format from S3
        if file_format == 'parquet':
            # Read parquet file from S3 using BytesIO
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            parquet_data = BytesIO(response['Body'].read())
            df = pd.read_parquet(parquet_data)
            
            log_message('info', f"Loaded parquet file with shape {df.shape}, columns: {list(df.columns)}")
            
            # Enhanced column mapping logic similar to standalone_rv_computation.py
            asset_type = config.get('asset_type', 'stocks')
            
            # Check for unexpected column structure and handle it
            if list(df.columns) == ['1', '2', '3', 'trades']:
                # This appears to be a parquet file with numeric column names
                # Map them to expected column names based on asset type
                if asset_type in ['forex', 'futures']:
                    # Forex/futures format: time, price, bid, ask, volume, trades
                    # But we only have 4 columns, so map to: time, price, bid, ask
                    df.columns = ['time', 'price', 'bid', 'ask']
                else:
                    # Stocks/ETFs format: time, price, volume, trades, is_not_outlier
                    # But we only have 4 columns, so map to: time, price, volume, trades
                    df.columns = ['time', 'price', 'volume', 'trades']
                
                log_message('info', f"Mapped parquet columns for ['1', '2', '3', 'trades']: {list(df.columns)}")
            elif len(df.columns) == 4 and all(str(col).isdigit() for col in df.columns):
                # Handle case with 4 numeric columns
                if asset_type in ['forex', 'futures']:
                    df.columns = ['time', 'price', 'bid', 'ask']
                else:
                    df.columns = ['time', 'price', 'volume', 'trades']
                
                log_message('info', f"Mapped 4-column numeric parquet: {list(df.columns)}")
            elif len(df.columns) == 5 and all(str(col).isdigit() for col in df.columns):
                # Handle case with 5 numeric columns
                if asset_type in ['forex', 'futures']:
                    df.columns = ['time', 'price', 'bid', 'ask', 'volume']
                else:
                    df.columns = ['time', 'price', 'volume', 'trades', 'is_not_outlier']
                
                log_message('info', f"Mapped 5-column numeric parquet: {list(df.columns)}")
            elif len(df.columns) == 5 and asset_type not in ['forex', 'futures']:
                # Handle stocks/ETFs with 5 columns - drop extra column
                df = df.iloc[:, :4]  # Take first 4 columns
                df.columns = ['time', 'price', 'volume', 'trades']
                log_message('warning', f"Dropped extra column from 5-column stocks/ETFs file: {list(df.columns)}")
            elif len(df.columns) == len(data_columns):
                # Standard case - rename columns
                df.columns = data_columns
                log_message('info', f"Applied standard column names: {data_columns}")
            elif 'time' in df.columns and 'price' in df.columns:
                # File has time and price columns but different structure
                # Keep original column names and just ensure we have required columns
                log_message('info', f"Using original column names with time and price: {list(df.columns)}")
            else:
                # Try to map columns intelligently based on position
                log_message('warning', f"Unexpected column structure in parquet file: {list(df.columns)}")
                if len(df.columns) >= 4:
                    # Map the most likely columns based on expected structure
                    if asset_type in ['forex', 'futures']:
                        df = df.iloc[:, :4]
                        df.columns = ['time', 'price', 'bid', 'ask']
                    else:
                        df = df.iloc[:, :4]
                        df.columns = ['time', 'price', 'volume', 'trades']
                    log_message('warning', f"Mapped columns intelligently: {list(df.columns)}")
                else:
                    log_message('error', f"Cannot process file with {len(df.columns)} columns")
                    return None
        else:
            # Read CSV (txt) file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            csv_data = StringIO(response['Body'].read().decode('utf-8'))
            df = pd.read_csv(csv_data, header=None, dtype=data_type)
            df.columns = data_columns
            log_message('info', f"Loaded CSV file with shape {df.shape}")

        # Ensure we have the required columns for processing
        required_cols = ['time', 'price']
        if not all(col in df.columns for col in required_cols):
            log_message('error', f"Missing required columns after processing. Available: {list(df.columns)}")
            return None

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
            log_message('info', "Added milliseconds to time column")
           
        # Convert time to datetime and set as index
        try:
            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
        except ValueError:
            # Try alternative formats
            try:
                df['time'] = pd.to_datetime(df['time'])
            except Exception as e:
                log_message('error', f"Failed to parse time column: {e}. Time sample: {df['time'].iloc[0] if len(df) > 0 else 'empty'}")
                return None
        
        df.set_index('time', inplace=True)
        log_message('info', f"Converted time column, index range: {df.index.min()} to {df.index.max()}")
        
        # Process price data based on asset type
        if config['asset_type'] in ['forex', 'futures'] and 'bid' in df.columns and 'ask' in df.columns:
            # For forex/futures with bid-ask data
            df['price'] = (df['bid'] + df['ask']) / 2
            log_message('info', "Calculated price from bid/ask for forex/futures")
        else:
            # For stocks and ETFs - adjust price based on outliers
            # if 'is_not_outlier' = 0 : overnight, if 'is_not_outlier' = NaN : outliers, if 'is_not_outlier' = 1 : good price
            if 'is_not_outlier' in df.columns:
                initial_count = len(df)
                # Filter out overnight data (is_not_outlier = 0)
                df = df[(df['is_not_outlier'] != 0) | df['is_not_outlier'].isna()]
                after_filter_count = len(df)
                log_message('info', f"Filtered overnight data: {initial_count} -> {after_filter_count} rows")
                
                # Apply outlier adjustment only where is_not_outlier is not NaN
                mask = df['is_not_outlier'].notna()
                df.loc[mask, 'price'] = df.loc[mask, 'price'] * df.loc[mask, 'is_not_outlier']
                log_message('info', "Applied is_not_outlier filter to price")
            else:
                # If is_not_outlier column doesn't exist, just use the price as is
                log_message('info', "No is_not_outlier column found, using price as-is")
                
        # Replace NaN values with local averages
        df = replace_nan(df)
        log_message('info', "Replaced NaN values with local averages")
        
        # Remove any remaining NaN values
        before_dropna = len(df)
        df = df.dropna(subset=['price'])
        after_dropna = len(df)
        log_message('info', f"Removed NaN prices: {before_dropna} -> {after_dropna} rows")
        
        # Sort by time to ensure proper ordering
        df.sort_index(inplace=True)
        
        if len(df) > 0:
            log_message('info', f"Final dataframe shape: {df.shape}, price range: {df['price'].min():.4f} to {df['price'].max():.4f}")
        else:
            log_message('warning', "Final dataframe is empty after processing")
        
        return df
        
    except Exception as e:
        log_message('error', f"Error loading data from {file_path}: {e}")
        import traceback
        log_message('error', f"Traceback: {traceback.format_exc()}")
        return None

def process_single_day(file_paths, assets, date, asset_configs, resample_freq='1T', 
                      resampling_method='last', measures=['RCov'], logger=None,
                      mixed_asset_mode=False, early_closing_day_file=None, s3_client=None, bucket_name=None):
    """
    Process a single day of data for covariance calculation.
    This function maintains the exact same logic as the original rcov_library.py
    """
    if not file_paths or len(file_paths) < 2:
        if logger:
            logger.warning(f"process_single_day: Insufficient file_paths for {date}: {len(file_paths) if file_paths else 0}")
        return None
    
    price_series = {}
    has_stocks = any(config['asset_type'] == 'stocks' for config in asset_configs.values())
    
    if logger:
        logger.info(f"process_single_day: Processing {date} with {len(assets)} assets, {len(file_paths)} files")
    
    # Load and process each asset's data
    for asset in assets:
        if asset not in file_paths:
            if logger:
                logger.warning(f"process_single_day: Asset {asset} not in file_paths for {date}")
            continue
            
        asset_config = asset_configs.get(asset, {})
        file_path = file_paths[asset]
        
        if logger:
            logger.info(f"process_single_day: Processing {asset} from {file_path}")
        
        try:
            # Load data using the prepare_data_rcov function
            df = prepare_data_rcov(file_path, asset_config, s3_client, bucket_name, logger)
            
            if df is None or df.empty:
                if logger:
                    logger.warning(f"process_single_day: No data loaded for {asset} on {date}")
                continue
                
            if len(df) < 10:
                if logger:
                    logger.warning(f"process_single_day: Insufficient data points for {asset} on {date}: {len(df)} < 10")
                continue
            
            if logger:
                logger.info(f"process_single_day: Loaded {len(df)} data points for {asset} on {date}")
            
            # Filter non-stock assets to stock trading hours when mixed mode
            if (mixed_asset_mode and has_stocks and 
                asset_config['asset_type'] in ['forex', 'futures']):
                
                # Early closing days hardcoded
                early_closing_days = [
                    '2015_11_27', '2015_12_24', '2016_11_25', '2016_12_24', '2017_07_03',
                    '2017_11_24', '2017_12_24', '2018_07_03', '2018_11_23', '2018_12_24',
                    '2019_07_03', '2019_11_29', '2019_12_24', '2020_11_27', '2020_12_24',
                    '2021_11_26', '2021_12_24', '2022_11_25', '2022_12_24', '2023_07_03',
                    '2023_11_24', '2023_12_24', '2024_07_03', '2024_11_29', '2024_12_24',
                    '2025_07_03', '2025_11_28', '2025_12_24', '2026_11_27', '2026_12_24',
                    '2027_11_26', '2027_12_24'
                ]
                
                # Simple trading hours filter for mixed assets
                day = date.replace('.txt', '').replace('.parquet', '')
                
                # Determine closing time
                if day in early_closing_days:
                    end_time = datetime.strptime('12:59:59.999', '%H:%M:%S.%f').time()
                else:
                    end_time = datetime.strptime('15:59:59.999', '%H:%M:%S.%f').time()
                
                start_time = datetime.strptime('09:30:00.000', '%H:%M:%S.%f').time()
                
                # Filter by trading hours
                time_mask = (df.index.time >= start_time) & (df.index.time <= end_time)
                df = df[time_mask]
                
                if len(df) < 10:
                    if logger:
                        logger.warning(f"process_single_day: Insufficient data after trading hours filter for {asset} on {date}: {len(df)} < 10")
                    continue
            
            # Use price column
            prices = df['price']
            
            if logger:
                logger.info(f"process_single_day: Resampling prices for {asset} on {date}: {len(prices)} points")
            
            # Resample prices
            resampled_prices, _ = resample_prices(prices, resample_freq, resampling_method)
            
            if len(resampled_prices) > 0:
                price_series[asset] = resampled_prices
                if logger:
                    logger.info(f"process_single_day: Successfully resampled {asset} on {date}: {len(resampled_prices)} points")
            else:
                if logger:
                    logger.warning(f"process_single_day: No resampled prices for {asset} on {date}")
                
        except Exception as e:
            if logger:
                logger.error(f"process_single_day: Error processing {asset} for {date}: {e}")
            continue
    
    if len(price_series) < 2:
        if logger:
            logger.warning(f"process_single_day: Insufficient price series for {date}: {len(price_series)} < 2")
        return None
    
    if logger:
        logger.info(f"process_single_day: Processing {len(price_series)} assets for {date}")
    
    try:
        # Maintain the order from the original asset configuration
        original_asset_order = list(asset_configs.keys())
        available_assets = [asset for asset in original_asset_order if asset in price_series]
        ordered_price_series = {asset: price_series[asset] for asset in available_assets}
        
        if logger:
            logger.info(f"process_single_day: Available assets for {date}: {available_assets}")
        
        sync_prices = pd.DataFrame(ordered_price_series).ffill()
        
        if logger:
            logger.info(f"process_single_day: Synchronized prices shape for {date}: {sync_prices.shape}")
        
        # Calculate returns
        returns = calculate_returns_from_prices(sync_prices)
        
        if len(returns) < 2:
            if logger:
                logger.warning(f"process_single_day: Insufficient returns for {date}: {len(returns)} < 2")
            return None
        
        if logger:
            logger.info(f"process_single_day: Returns shape for {date}: {returns.shape}")
            
        # Calculate covariance matrices
        cov_matrices = calculate_covariance_matrices(returns, measures)
        
        if logger:
            logger.info(f"process_single_day: Calculated {len(cov_matrices)} covariance matrices for {date}")
            for measure_name, matrix in cov_matrices.items():
                logger.info(f"process_single_day: {measure_name} matrix shape: {matrix.shape}")
        
        return cov_matrices
        
    except Exception as e:
        if logger:
            logger.error(f"process_single_day: Error calculating covariance for {date}: {e}")
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

# ============================================================================
# S3 DATA PROCESSOR CLASS
# ============================================================================

class S3CovarianceProcessor:
    """S3 Data Processor for Realized Covariance Computation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.s3_client = self._setup_s3_client()
        self.db_config = self._setup_db_config()
        
        # Setup bucket discovery
        self.bucket_name = self._discover_bucket()
        if not self.bucket_name:
            raise ValueError("No accessible S3 bucket found")
            
        # Date pattern for file parsing
        self.date_pattern = re.compile(r'(\d{4})_(\d{2})_(\d{2})')
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('standalone_rcov')
        logger.setLevel(logging.INFO)
        return logger
        
    def _setup_s3_client(self):
        """Setup S3 client with Windmill or environment variables"""
        try:
            if WINDMILL_AVAILABLE:
                # Get S3 credentials from Windmill
                s3_endpoint_url = wmill.get_variable("u/niccolosalvini27/S3_ENDPOINT_URL") or 'http://localhost:9000'
                s3_access_key = wmill.get_variable("u/niccolosalvini27/S3_ACCESS_KEY") or 'minioadmin'
                s3_secret_key = wmill.get_variable("u/niccolosalvini27/S3_SECRET_KEY") or 'minioadmin'
            else:
                # Fallback to environment variables for local testing
                s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:9000')
                s3_access_key = os.getenv('S3_ACCESS_KEY', 'minioadmin')
                s3_secret_key = os.getenv('S3_SECRET_KEY', 'minioadmin')
            
            s3_client = boto3.client(
                's3',
                endpoint_url=s3_endpoint_url,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                region_name='us-east-1'
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
    
    def _setup_db_config(self):
        """Setup database connection configuration"""
        try:
            if WINDMILL_AVAILABLE:
                # Get database credentials from Windmill
                db_config = {
                    'host': wmill.get_variable("u/niccolosalvini27/DB_HOST") or 'volare.unime.it',
                    'port': int(wmill.get_variable("u/niccolosalvini27/DB_PORT") or 5432),
                    'database': wmill.get_variable("u/niccolosalvini27/DB_NAME") or 'forvarddb',
                    'user': wmill.get_variable("u/niccolosalvini27/DB_USER") or 'forvarduser',
                    'password': wmill.get_variable("u/niccolosalvini27/DB_PASSWORD") or 'WsUpwXjEA7HHidmL8epF'
                }
            else:
                # Fallback to environment variables
                db_config = {
                    'host': os.getenv('DB_HOST', 'forvard_app_postgres'),
                    'port': int(os.getenv('DB_PORT', 5432)),
                    'database': os.getenv('DB_NAME', 'forvarddb'),
                    'user': os.getenv('DB_USER', 'forvarduser'),
                    'password': os.getenv('DB_PASSWORD', 'WsUpwXjEA7HHidmL8epF')
                }
            
            self.logger.info(f"Database config: {db_config['host']}:{db_config['port']}/{db_config['database']}")
            return db_config
            
        except Exception as e:
            self.logger.error(f"Failed to setup database config: {e}")
            raise
    
    def _discover_bucket(self) -> Optional[str]:
        """Discover available S3 bucket"""
        try:
            # Get bucket name from Windmill or environment
            if WINDMILL_AVAILABLE:
                bucket_name = wmill.get_variable("u/niccolosalvini27/S3_BUCKET")
            else:
                bucket_name = os.getenv('S3_BUCKET')
            
            if bucket_name:
                try:
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    self.logger.info(f"Using configured bucket: {bucket_name}")
                    return bucket_name
                except Exception as e:
                    self.logger.warning(f"Configured bucket {bucket_name} not accessible: {e}")
            
            # Try common bucket names if configured one fails
            common_buckets = ['datalake-dev', 'datalake', 'data', 'forvard-data']
            for bucket in common_buckets:
                try:
                    self.s3_client.head_bucket(Bucket=bucket)
                    self.logger.info(f"Using discovered bucket: {bucket}")
                    return bucket
                except:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Bucket discovery failed: {e}")
            return None
    
    def find_files_for_symbols_and_date(self, symbols: List[str], date: str) -> Dict[str, str]:
        """Find S3 files for all symbols on a specific date"""
        file_paths = {}
        
        self.logger.info(f"find_files_for_symbols_and_date: Looking for files for date {date} and {len(symbols)} symbols")
        
        for symbol in symbols:
            symbol_config = self.config['symbols'][symbol]
            asset_type = symbol_config['asset_type']
            file_format = symbol_config.get('file_format', 'parquet')
            
            # Use the S3 path from configuration if available, otherwise construct it
            if 's3_path' in symbol_config:
                s3_prefix = f"{symbol_config['s3_path']}/"
            else:
                # Fallback to old method
                s3_prefix = f"data/{asset_type}/{symbol}/"
            
            try:
                self.logger.info(f"Searching S3 path: s3://{self.bucket_name}/{s3_prefix}")
                
                # Use paginator to handle S3 pagination (more than 1000 files)
                files = []
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            filename = key.split('/')[-1]
                            
                            # Filter out unwanted files
                            if filename.endswith(('.txt', '.parquet')):
                                if not filename.endswith('_last_update.txt') and filename != 'adjustment.txt':
                                    files.append(key)
                
                self.logger.info(f"Found {len(files)} files for {symbol}")
                
                # Find file for specific date
                # Files are stored as YYYY_MM_DD.parquet, not SYMBOL_YYYY_MM_DD.parquet
                target_filename = f"{date}.{file_format}"
                self.logger.info(f"Looking for target file: {target_filename}")
                
                # Log first few files to help with debugging
                if len(files) > 0:
                    sample_files = [f.split('/')[-1] for f in files[:5]]
                    self.logger.info(f"Sample filenames for {symbol}: {sample_files}")
                
                found_file = False
                for file_key in files:
                    if file_key.endswith(target_filename):
                        file_paths[symbol] = file_key
                        self.logger.info(f"Found target file for {symbol}: {file_key}")
                        found_file = True
                        break
                
                if not found_file:
                    self.logger.warning(f"Target file {target_filename} not found for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error searching files for {symbol}: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        self.logger.info(f"find_files_for_symbols_and_date: Found files for {len(file_paths)} out of {len(symbols)} symbols for date {date}")
        return file_paths
    
    def get_date_range_from_config(self) -> Tuple[datetime, datetime]:
        """Get date range from configuration"""
        start_date_str = self.config['processing'].get('start_date')
        end_date_str = self.config['processing'].get('end_date')
        
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y_%m_%d')
        else:
            start_date = datetime(2024, 3, 1)  # Default start
        
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y_%m_%d')
        else:
            end_date = datetime(2024, 3, 31)  # Default end
        
        return start_date, end_date
    
    def generate_date_list(self) -> List[str]:
        """Generate list of dates to process"""
        start_date, end_date = self.get_date_range_from_config()
        
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (assuming market data)
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                dates.append(current_date.strftime('%Y_%m_%d'))
            current_date += timedelta(days=1)
        
        return dates
    
    def process_single_date(self, date: str) -> Optional[List[Dict[str, Any]]]:
        """Process covariance calculation for a single date"""
        try:
            symbols = list(self.config['symbols'].keys())
            
            # Find files for all symbols on this date
            file_paths = self.find_files_for_symbols_and_date(symbols, date)
            
            if len(file_paths) < 2:
                self.logger.warning(f"Insufficient data for {date}: only {len(file_paths)} symbols found")
                return None
            
            # Get processing configuration
            measures = self.config['processing'].get('measures', ['RCov', 'RBPCov', 'RSCov'])
            resample_freq = self.config['processing'].get('resample_freq', '1T')  # Use pandas time format
            resampling_method = self.config['processing'].get('resampling_method', 'last')
            
            # Process the day
            cov_matrices = process_single_day(
                file_paths=file_paths,
                assets=symbols,
                date=date,
                asset_configs=self.config['symbols'],
                resample_freq=resample_freq,
                resampling_method=resampling_method,
                measures=measures,
                logger=self.logger,
                mixed_asset_mode=False,
                early_closing_day_file=None,
                s3_client=self.s3_client,
                bucket_name=self.bucket_name
            )
            
            if not cov_matrices:
                self.logger.warning(f"No covariance matrices computed for {date}")
                return None
            
            # Format output - this creates rows with 'date' field
            formatted_df = format_covariance_output(cov_matrices, date)
            
            # Convert to list of dictionaries for database insertion
            results = []
            for _, row in formatted_df.iterrows():
                # Determine asset_type from the asset configuration
                asset1_type = self.config['symbols'].get(row['asset1'], {}).get('asset_type', 'unknown')
                asset2_type = self.config['symbols'].get(row['asset2'], {}).get('asset_type', 'unknown')
                
                # Debug logging
                self.logger.info(f"Debug: asset1={row['asset1']}, asset1_type={asset1_type}")
                self.logger.info(f"Debug: asset2={row['asset2']}, asset2_type={asset2_type}")
                self.logger.info(f"Debug: Available symbols in config: {list(self.config['symbols'].keys())[:5]}...")
                
                # For mixed assets, use the first asset's type as the primary type
                # In practice, both assets should be of the same type in a single computation
                asset_type = asset1_type
                
                result_dict = {
                    'date': row['date'],  # Map 'date' to 'date'
                    'asset1': row['asset1'],   # Map 'asset1' to 'asset1'
                    'asset2': row['asset2'],   # Map 'asset2' to 'asset2'
                    'asset_type': asset_type,  # Add asset_type
                }
                
                # Add covariance measures with proper column mapping
                if 'RCov' in row:
                    result_dict['rcov'] = row['RCov']
                if 'RBPCov' in row:
                    result_dict['rbpcov'] = row['RBPCov']
                if 'RSCov_P' in row:
                    result_dict['rscov_p'] = row['RSCov_P']
                if 'RSCov_N' in row:
                    result_dict['rscov_n'] = row['RSCov_N']
                if 'RSCov_Mp' in row:
                    result_dict['rscov_mp'] = row['RSCov_Mp']
                if 'RSCov_Mn' in row:
                    result_dict['rscov_mn'] = row['RSCov_Mn']
                
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing date {date}: {e}")
            return None
    
    def save_to_database(self, results: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Save covariance results to PostgreSQL database with proper transaction handling"""
        if not results:
            return 0, 0
        
        def convert_numpy_types(value):
            """Convert numpy types to Python native types"""
            if value is None:
                return None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value
        
        skipped_count = 0
        inserted_count = 0
        
        # Process each record in its own transaction to avoid transaction abort issues
        for result in results:
            conn = None
            cursor = None
            try:
                # Connect to database for each record (autocommit mode)
                conn = psycopg2.connect(**self.db_config)
                conn.autocommit = True  # This prevents transaction abort issues
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # Convert numpy types to Python native types
                converted_result = {k: convert_numpy_types(v) for k, v in result.items()}
                
                # Debug logging for database insertion
                self.logger.info(f"Debug DB: Inserting record with keys: {list(converted_result.keys())}")
                self.logger.info(f"Debug DB: asset_type = '{converted_result.get('asset_type', 'NOT_FOUND')}'")
                
                # Check if record already exists
                check_query = """
                    SELECT id FROM realized_covariance_data 
                    WHERE date = %s AND asset1 = %s AND asset2 = %s AND asset_type = %s
                """
                cursor.execute(check_query, (
                    converted_result['date'], 
                    converted_result['asset1'], 
                    converted_result['asset2'],
                    converted_result['asset_type']
                ))
                existing = cursor.fetchone()
                
                if existing:
                    skipped_count += 1
                    continue
                
                # Insert new record
                insert_query = """
                    INSERT INTO realized_covariance_data (
                        date, asset1, asset2, asset_type,
                        rcov, rbpcov, rscov_p, rscov_n, rscov_mp, rscov_mn
                    ) VALUES (
                        %(date)s, %(asset1)s, %(asset2)s, %(asset_type)s,
                        %(rcov)s, %(rbpcov)s, %(rscov_p)s, %(rscov_n)s, 
                        %(rscov_mp)s, %(rscov_mn)s
                    )
                """
                
                cursor.execute(insert_query, converted_result)
                inserted_count += 1
                
            except Exception as e:
                self.logger.error(f"Error inserting record for {result.get('asset1', 'unknown')}-{result.get('asset2', 'unknown')} {result.get('date', 'unknown')}: {e}")
                # With autocommit, no need to rollback - each operation is atomic
                continue
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
        
        self.logger.info(f"Database save completed: {inserted_count} inserted, {skipped_count} skipped")
        return skipped_count, inserted_count

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def process_dates_batch(processor: S3CovarianceProcessor, dates: List[str], max_workers: int = 2) -> Dict[str, Any]:
    """Process multiple dates in parallel with limited concurrency"""
    results = {
        "total_dates": len(dates),
        "processed": 0,
        "errors": 0,
        "total_inserted": 0,
        "total_skipped": 0
    }
    
    # Process dates with limited concurrency to avoid overwhelming S3
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_date = {
            executor.submit(processor.process_single_date, date): date 
            for date in dates
        }
        
        # Process results as they complete
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            
            try:
                date_results = future.result()
                
                if date_results:
                    # Save to database
                    skipped, inserted = processor.save_to_database(date_results)
                    results["processed"] += 1
                    results["total_inserted"] += inserted
                    results["total_skipped"] += skipped
                    logger.info(f"Date {date}: {inserted} inserted, {skipped} skipped")
                else:
                    results["errors"] += 1
                    logger.warning(f"Date {date}: No results")
                
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Date {date}: Error - {e}")
            
            # Add small delay to reduce S3 pressure
            time.sleep(0.1)
        
        # Progress update
        if (results["processed"] + results["errors"]) % 10 == 0:
            logger.info(f"Progress: {results['processed'] + results['errors']}/{len(dates)} dates processed")
    
    return results

def main(
    # Pipeline configuration
    pipeline_name="Realized Covariance",
    pipeline_enabled=True,
    
    # Asset type configuration
    stocks_enabled=True,
    stocks_symbols=[
        "MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY", "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA", "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM", "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK", "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC"
    ],
    forex_enabled=True,
    forex_symbols=[
        "EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "JPYUSD", "CHFUSD", "SGDUSD", "HKDUSD", "KRWUSD", "INRUSD", "RUBUSD", "BRLUSD"
    ],
    futures_enabled=True,
    futures_symbols=[
        "ES", "CL", "GC", "NG", "NQ", "TY", "FV", "EU", "SI", "C", "W", "VX"
    ],
    etfs_enabled=True,
    etfs_symbols=[
        "AGG", "BND", "GLD", "SLV", "SUSA", "EFIV", "ESGV", "ESGU", "AFTY", "MCHI", "EWH", "EEM", "IEUR", "VGK", "FLCH", "EWJ", "NKY", "EWZ", "EWC", "EWU", "EWI", "EWP", "ACWI", "IOO", "GWL", "VEU", "IJH", "MDY", "IVOO", "IYT", "XTN", "XLI", "XLU", "VPU", "SPSM", "IJR", "VIOO", "QQQ", "ICLN", "ARKK", "SPLG", "SPY", "VOO", "IYY", "VTI", "DIA"
    ],
    mixed_enabled=False,
    mixed_symbols=["GE:stocks", "JNJ:stocks", "EURUSD:forex", "JPYUSD:forex", "CL:futures", "GC:futures"],
    
    # Date range
    start_date="2015_01_01",
    end_date="2025_07_17",
    
    # Processing settings
    resample_freq="1T",
    resampling_method="last",
    measures=None,
    max_workers=8,
    batch_size=50,
    
    # File format
    file_format="parquet"
):
    """
    Standalone Realized Covariance Computation
    
    Args:
        pipeline_name: Name of the pipeline (e.g., 'futures', 'stocks', 'forex', 'mixed_assets')
        pipeline_enabled: Whether the pipeline is enabled
        
        # Asset type configuration
        stocks_enabled: Enable stocks pipeline
        stocks_symbols: List of stock symbols
        forex_enabled: Enable forex pipeline  
        forex_symbols: List of forex symbols
        futures_enabled: Enable futures pipeline
        futures_symbols: List of futures symbols
        mixed_enabled: Enable mixed assets pipeline
        mixed_symbols: List of mixed symbols with format "SYMBOL:TYPE"
        
        start_date: Start date in MM/DD/YYYY format
        end_date: End date in MM/DD/YYYY format
        resample_freq: Resampling frequency (e.g., '1T', '5T')
        resampling_method: Resampling method ('last', 'mean', 'first')
        measures: List of covariance measures to compute
        max_workers: Maximum number of parallel workers
        batch_size: Number of dates to process in each batch
        file_format: File format ('parquet' or 'txt')
    """
    # Early closing days (hardcoded as requested)
    early_closing_days = [
        '2015_11_27', '2015_12_24', '2016_11_25', '2016_12_24', '2017_07_03',
        '2017_11_24', '2017_12_24', '2018_07_03', '2018_11_23', '2018_12_24',
        '2019_07_03', '2019_11_29', '2019_12_24', '2020_11_27', '2020_12_24',
        '2021_11_26', '2021_12_24', '2022_11_25', '2022_12_24', '2023_07_03',
        '2023_11_24', '2023_12_24', '2024_07_03', '2024_11_29', '2024_12_24',
        '2025_07_03', '2025_11_28', '2025_12_24', '2026_11_27', '2026_12_24',
        '2027_11_26', '2027_12_24'
    ]
    
    # Determine which asset type to process based on enabled flags
    if stocks_enabled:
        asset_type = "stocks"
        symbols_list = stocks_symbols
        s3_path_prefix = "data/stocks"
    elif forex_enabled:
        asset_type = "forex"
        symbols_list = forex_symbols
        s3_path_prefix = "data/forex"
    elif futures_enabled:
        asset_type = "futures"
        symbols_list = futures_symbols
        s3_path_prefix = "data/futures"
    elif etfs_enabled:
        asset_type = "etfs"
        symbols_list = etfs_symbols
        s3_path_prefix = "data/ETFs"  # Note: ETFs uses capital letters in S3
    elif mixed_enabled:
        asset_type = "mixed"
        symbols_list = mixed_symbols
        s3_path_prefix = "data"  # Mixed assets can be in different subdirectories
    else:
        # Default to futures if nothing is enabled
        asset_type = "futures"
        symbols_list = futures_symbols
        s3_path_prefix = "data/futures"
    
    # Log the S3 path prefix for verification
    logger.info(f"Asset type: {asset_type}")
    logger.info(f"S3 path prefix: {s3_path_prefix}")
    logger.info(f"Symbols to process: {len(symbols_list)}")
    
    # Build symbols dictionary based on asset type and symbols list
    symbols = {}
    
    if asset_type == "mixed":
        # Parse mixed symbols with format "SYMBOL:TYPE"
        for symbol_entry in symbols_list:
            if ':' in symbol_entry:
                symbol, symbol_type = symbol_entry.split(':', 1)
                symbols[symbol] = {
                    "asset_type": symbol_type.lower(),
                    "file_format": file_format,
                    "s3_path": f"data/{symbol_type.lower()}/{symbol}"
                }
            else:
                # Default to stocks if no type specified
                symbols[symbol_entry] = {
                    "asset_type": "stocks",
                    "file_format": file_format,
                    "s3_path": f"data/stocks/{symbol_entry}"
                }
    else:
        # Single asset type
        for symbol in symbols_list:
            symbols[symbol] = {
                "asset_type": asset_type,
                "file_format": file_format,
                "s3_path": f"{s3_path_prefix}/{symbol}"
            }
    
    # Default measures if none provided
    if measures is None:
        measures = ["RCov", "RBPCov", "RSCov"]
    
    # Convert dates from MM/DD/YYYY to YYYY_MM_DD format
    try:
        start_date_converted = datetime.strptime(start_date, '%m/%d/%Y').strftime('%Y_%m_%d')
        end_date_converted = datetime.strptime(end_date, '%m/%d/%Y').strftime('%Y_%m_%d')
    except ValueError:
        # If dates are already in YYYY_MM_DD format, use as is
        start_date_converted = start_date
        end_date_converted = end_date
    
    # Configuration structure matching the original format
    config = {
        "general": {
            "file_format": file_format,
            "rv_threads_max": max_workers
        },
        "covariance_settings": {
            "resample_freq": resample_freq,
            "resampling_method": resampling_method
        },
        "symbols": symbols,
        "processing": {
            "start_date": start_date_converted,
            "end_date": end_date_converted,
            "measures": measures,
            "resample_freq": resample_freq,
            "resampling_method": resampling_method,
            "max_workers": max_workers,
            "batch_size": batch_size
        }
    }
    
    start_time = datetime.now()
    logger.info(f"RCOV COMPUTATION START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"Asset type: {asset_type}")
    logger.info(f"Processing symbols: {list(config['symbols'].keys())}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Log which asset types are enabled
    enabled_types = []
    if stocks_enabled:
        enabled_types.append(f"stocks({len(stocks_symbols)} symbols)")
    if forex_enabled:
        enabled_types.append(f"forex({len(forex_symbols)} symbols)")
    if futures_enabled:
        enabled_types.append(f"futures({len(futures_symbols)} symbols)")
    if etfs_enabled:
        enabled_types.append(f"etfs({len(etfs_symbols)} symbols)")
    if mixed_enabled:
        enabled_types.append(f"mixed({len(mixed_symbols)} symbols)")
    
    if enabled_types:
        logger.info(f"Enabled asset types: {', '.join(enabled_types)}")
    else:
        logger.info("No asset types enabled, using default (futures)")
    
    computation_start_time = time.time()
    
    try:
        # Initialize processor
        processor = S3CovarianceProcessor(config)
        
        # Generate date list
        dates = processor.generate_date_list()
        logger.info(f"Processing {len(dates)} dates from {dates[0] if dates else 'none'} to {dates[-1] if dates else 'none'}")
        
        if not dates:
            logger.error("No dates to process")
            return
        
        # Process all dates
        max_workers = config['processing'].get('max_workers', 2)
        batch_size = config['processing'].get('batch_size', 50)
        
        # Process in batches to manage memory and connections
        all_results = {
            "total_dates": len(dates),
            "processed": 0,
            "errors": 0,
            "total_inserted": 0,
            "total_skipped": 0
        }
        
        for i in range(0, len(dates), batch_size):
            batch_dates = dates[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: dates {i+1}-{min(i+batch_size, len(dates))}")
            
            batch_results = process_dates_batch(processor, batch_dates, max_workers)
            
            # Aggregate results
            all_results["processed"] += batch_results["processed"]
            all_results["errors"] += batch_results["errors"]
            all_results["total_inserted"] += batch_results["total_inserted"]
            all_results["total_skipped"] += batch_results["total_skipped"]
            
            logger.info(f"Batch completed: {batch_results['processed']} processed, {batch_results['errors']} errors")
        
        # Final results
        execution_time = time.time() - computation_start_time
        success_rate = (all_results["processed"] / all_results["total_dates"]) * 100 if all_results["total_dates"] > 0 else 0
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("STANDALONE RCOV COMPUTATION COMPLETED")
        logger.info(f"Pipeline: {pipeline_name} ({asset_type})")
        logger.info(f"Total dates: {all_results['total_dates']}")
        logger.info(f"Successfully processed: {all_results['processed']}")
        logger.info(f"Errors: {all_results['errors']}")
        logger.info(f"Records inserted: {all_results['total_inserted']}")
        logger.info(f"Records skipped (duplicates): {all_results['total_skipped']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info("=" * 60)
        
        # Prepare Slack notification
        if all_results["processed"] > 0:
            status_emoji = "✅"
            status_text = "SUCCESS"
        else:
            status_emoji = "❌"
            status_text = "FAILURE"
        
        # Create Slack message
        slack_message = f"{status_emoji} *RCOV COMPUTATION COMPLETED* {status_emoji}\n\n"
        slack_message += f"*Status:* {status_text}\n"
        slack_message += f"*Pipeline:* {pipeline_name}\n"
        slack_message += f"*Asset Type:* {asset_type}\n"
        slack_message += f"*Duration:* {str(total_duration).split('.')[0]}\n"
        slack_message += f"*Symbols:* {len(symbols)}\n\n"
        
        # Add enabled asset types info
        enabled_types = []
        if stocks_enabled:
            enabled_types.append(f"stocks({len(stocks_symbols)})")
        if forex_enabled:
            enabled_types.append(f"forex({len(forex_symbols)})")
        if futures_enabled:
            enabled_types.append(f"futures({len(futures_symbols)})")
        if etfs_enabled:
            enabled_types.append(f"etfs({len(etfs_symbols)})")
        if mixed_enabled:
            enabled_types.append(f"mixed({len(mixed_symbols)})")
        
        if enabled_types:
            slack_message += f"*Enabled Types:* {', '.join(enabled_types)}\n\n"
        
        slack_message += "*Processing Summary:*"
        slack_message += f"\n📅 Total dates: {all_results['total_dates']}"
        slack_message += f"\n✅ Successfully processed: {all_results['processed']}"
        slack_message += f"\n❌ Errors: {all_results['errors']}"
        slack_message += f"\n💾 Records inserted: {all_results['total_inserted']}"
        slack_message += f"\n⏭️ Records skipped: {all_results['total_skipped']}"
        
        if all_results['total_dates'] > 0:
            slack_message += f"\n📊 Success rate: {success_rate:.1f}%"
        
        slack_message += f"\n\n**Timestamp:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send Slack notification
        try:
            send_slack_notification(slack_message)
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
