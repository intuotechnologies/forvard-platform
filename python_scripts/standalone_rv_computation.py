"""
ForVARD Project - Standalone Realized Variance Computation
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 29/05/2025

2025 University of Messina, Department of Economics.
Research code - Unauthorized distribution prohibited.

Standalone script that:
1. Reads processed tick data from S3 datalake
2. Computes realized variance measures using identical logic to main_pipeline
3. Saves results to PostgreSQL database in the forvard network
"""

import os
import sys
import time
import logging
import argparse
import threading
import traceback
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import re
import pyarrow

# S3/MinIO imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Try to import wmill, use fallback if not available
import requests
import wmill
WINDMILL_AVAILABLE = True

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
# EMBEDDED REALIZED VARIANCE LIBRARY (from rv_library.py)
# ============================================================

def parkinson_variance(data):
    """Calculate Parkinson's Range-based Volatility Estimator for High-Frequency Data"""
    prices = (data['price'])
  
    if len(prices) < 2:
        return np.nan
    
    high_price = prices.max()
    low_price = prices.min()
    
    # Parkinson's volatility formula
    ln_hl_ratio = np.log(high_price / low_price)
    pv = ((1 / (4 * np.log(2))) * ln_hl_ratio**2)
    
    return high_price, low_price, pv

def garman_klass_variance(data, asset_type):
    """Calculate Garman-Klass Range-based Volatility Estimator for High-Frequency Data"""
    prices = data['price'].values

    if len(data) < 2:
        return np.nan 
    high_price = prices.max()
    low_price = prices.min()

    if asset_type=='stocks' or asset_type=='ETFs':
        filtered_df = data[data['volume'] >= 100]
        open_price = filtered_df['price'].iloc[0]   
        close_price = filtered_df['price'].iloc[-1] 
    else:
        open_price = data['price'].iloc[0]
        close_price = data['price'].iloc[-1]
    
    # Calculate volatility
    ln_hl_ratio = np.log(high_price / low_price)
    ln_co_ratio = np.log(close_price/  open_price)
    
    gkv = (1/2) * (ln_hl_ratio**2) - (2 * np.log(2) - 1) * (ln_co_ratio**2)
      
    return open_price, close_price, gkv 

def realized_range(data):
    """Calculate the realized range proposed by Martens and Van Dijk (2007) and Christensen and Podolskij (2007)"""
    prices = (data['price'])
        
    # Resample the data to the desired frequency
    # We need to delete the first observation which set instead of nan the first value of the series
    high = np.log(prices.resample('5min', label='right', closed='right').max())[1:]
    low = np.log(prices.resample('5min', label='right', closed='right').min())[1:]
  
    # Calculate the realized variance range for each period
    scaling_factor = 1/( 4 * np.log(2) )# Scale by the theoretical constant (4 log 2)
    realized_range = scaling_factor * np.sum((high - low) ** 2) 
    
    return realized_range

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

def resample_prices(prices, resample_freq='5min', resampling_method='last', origin_offset_minutes=0):
    """Unified resampling with SMART backfill logic - no excessive backfill for forex."""
    
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

def calculate_subsampled_metric(prices, resample_freq, metric_func, n_subsamples=5, resampling_method='last'):
    """Calculate subsampled metric with consistent backfill logic."""
    # Parse frequency to determine offset step
    base_freq = int(''.join(filter(str.isdigit, resample_freq)))
    offset_step = base_freq / n_subsamples  # minutes per subsample
    
    subsample_metrics = []
    subsample_M_values = []
    
    # Calculate all subsamples (including standard as subsample 0)
    for i in range(n_subsamples):
        offset_minutes = offset_step * i
        
        try:
            resampled_i, M_i = resample_prices(
                prices, resample_freq, resampling_method, 
                origin_offset_minutes=offset_minutes
            )
            
            if len(resampled_i) >= 2 and M_i >= 1:
                subsample_metric = metric_func(resampled_i, M_i)
                if not np.isnan(subsample_metric):
                    subsample_metrics.append(subsample_metric)
                    subsample_M_values.append(M_i)
                        
        except Exception as e:
            print(f"Error processing subsample {i}: {e}")
            continue

    if len(subsample_metrics) == 0:
        return np.nan, np.nan
    
    # First metric is standard, average is subsampled
    standard_metric = subsample_metrics[0]
    subsampled_metric = np.mean(subsample_metrics)
    
    return standard_metric, subsampled_metric

def realized_power_variation(data, exp=2, resample_freq='5min', price_col='price', 
                      resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """Calculate Realized Variance (exp=2) or Realized Quarticity (exp=4) with subsampling."""
    
    def realized_power_variation_metric(resampled_prices, M, exp=2):
        """Calculate realized power variation from resampled prices."""
        log_returns = np.log(resampled_prices).diff().dropna()
        rv = (log_returns ** exp).sum()
        
        if exp == 4:
            return (M/3) * rv
        else:
            return rv
    
    if exp not in [2, 4]:
        raise ValueError("exp must be 2 (variance) or 4 (quarticity)")
    
    # Ensure index is datetime and sorted
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    prices = data[price_col]
    
    # Return early if subsampling not requested
    if not calculate_subsample:
        resampled, M = resample_prices(prices, resample_freq, resampling_method)
        rv = realized_power_variation_metric(resampled, M, exp)
        return rv
    else:
        # Subsampling
        def metric_func_wrapper(resampled_prices, M):
            return realized_power_variation_metric(resampled_prices, M, exp)
        
        rv, rv_ss = calculate_subsampled_metric(
            prices, resample_freq, metric_func_wrapper, n_subsamples, resampling_method
        )
        return rv, rv_ss

def bipower_variation(data, resample_freq='5min', price_col='price', 
                      resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """Calculate Bipower Variation with subsampling."""

    def bipower_variation_metric(resampled_prices, M = None):
        """Calculate bipower variation from resampled prices."""
        log_returns = np.log(resampled_prices).diff().dropna()
        lagged_returns = log_returns.shift(1)
        return_products = np.abs(log_returns) * np.abs(lagged_returns)
        return_products = return_products.dropna()
        
        bv = (np.pi / 2) * np.sum(return_products)
        
        return bv

    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {e}")
    
    # Ensure data is sorted by time
    data = data.sort_index()

    # Extract price series
    prices = data[price_col]

    # Calculate standard bipower variation
    if not calculate_subsample:
        resampled, M = resample_prices(prices, resample_freq, resampling_method) 
        bv = bipower_variation_metric(resampled, M)
        return bv
    else:
        bv, bv_ss = calculate_subsampled_metric(
            prices, 
            resample_freq, 
            bipower_variation_metric,
            n_subsamples, 
            resampling_method)        
        return bv, bv_ss

def realized_semivariance(data, resample_freq='5min', price_col='price', 
                         resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """Calculate Realized Semi-variance (RSV) and optionally Subsampled RSV (RSV_SS)"""
    def realized_positive_semivariance_metric(resampled_prices, M=None):
        """Calculate realized semivariance (positive and negative) from resampled prices."""
        # Calculate log returns
        log_returns = np.log(resampled_prices).diff().dropna()
        rsv_positive = (log_returns[log_returns > 0] ** 2).sum()
        return rsv_positive
    
    def realized_negative_semivariance_metric(resampled_prices, M=None):
        """Calculate realized semivariance (positive and negative) from resampled prices."""
        # Calculate log returns
        log_returns = np.log(resampled_prices).diff().dropna()
        rsv_negative = (log_returns[log_returns < 0] ** 2).sum()
        return rsv_negative

    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {e}")
    
    # Ensure data is sorted by time
    data = data.sort_index()

    # Extract price series
    prices = data[price_col]   
    
    # Return early if subsampling is not requested
    if not calculate_subsample:
        # Calculate standard realized semivariance
        resampled, M = resample_prices(prices, resample_freq, resampling_method)  
        rsv_positive = realized_positive_semivariance_metric(resampled, M)
        rsv_negative = realized_negative_semivariance_metric(resampled, M)
        return rsv_positive, rsv_negative
    else:    
        rsv_positive, rsv_ss_positive = calculate_subsampled_metric(
            prices, 
            resample_freq, 
            realized_positive_semivariance_metric,
            n_subsamples, 
            resampling_method
        )
        
        rsv_negative, rsv_ss_negative = calculate_subsampled_metric(
            prices, 
            resample_freq, 
            realized_negative_semivariance_metric,
            n_subsamples, 
            resampling_method
        )
   
        return rsv_positive, rsv_negative, rsv_ss_positive, rsv_ss_negative

def median_realized_variance(data, resample_freq='5min', price_col='price', 
                            resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """Calculate Median Realized Variance (MedRV) with subsampling."""
    def median_rv_metric(resampled_prices, M=None):
        """Calculate median realized variance from resampled prices."""
        log_returns = np.diff(np.log(resampled_prices.values))
        
        # Calculate absolute returns
        abs_returns = np.abs(log_returns)
        
        n = len(abs_returns)
        
        # If we don't have enough returns, return NaN
        if n < 3:
            return np.nan
        
        # Calculate MedRV
        rr = np.array([abs_returns[0:n-2], abs_returns[1:n-1], abs_returns[2:n]]).T
        medRV = np.sum(np.median(rr, axis=1)**2)
        
        # Apply scaling factor
        medRVScale = np.pi/(6-4*np.sqrt(3)+np.pi)
        medRV = medRV * medRVScale * (M/(M-2))
        
        return medRV
        
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {e}")
    # Ensure data is sorted by time
    data = data.sort_index()

    # Extract price series
    prices = data[price_col]
    
    # Calculate standard median realized variance
    # Return early if subsampling is not requested
    if not calculate_subsample:
        resampled, M = resample_prices(prices, resample_freq, resampling_method)
        medrv = median_rv_metric(resampled, M)  
        return medrv
    else:
        # Use the subsampling utility
        medrv, medrv_ss = calculate_subsampled_metric(
            prices, 
            resample_freq, 
            median_rv_metric, 
            n_subsamples, 
            resampling_method
        )

        return medrv, medrv_ss

def min_realized_variance(data, resample_freq='5min', price_col='price', 
                         resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """Calculate Minimum Realized Variance (MinRV) with subsampling."""
    def min_rv_metric(resampled_prices, M=None):
        """Calculate minimum realized variance from resampled prices."""
        log_returns = np.diff(np.log(resampled_prices.values))
        
        # Calculate absolute returns
        abs_returns = np.abs(log_returns)
        
        n = len(abs_returns)
        
        # If we don't have enough returns, return NaN
        if n < 2:
            return np.nan
        
        # Calculate MinRV
        rr = np.array([abs_returns[0:n-1], abs_returns[1:n]]).T
        minRV = np.sum(np.min(rr, axis=1)**2)
        
        # Apply scaling factor
        minRVscale = np.pi/(np.pi-2)
        minRV = minRV * minRVscale * (M/(M-2))
        
        return minRV
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {e}")
    
    # Ensure data is sorted by time
    data = data.sort_index()

    # Extract price series
    prices = data[price_col]
    
    # Calculate standard minimum realized variance    
    # Return early if subsampling is not requested
    if not calculate_subsample:
        resampled, M = resample_prices(prices, resample_freq, resampling_method)
        minrv = min_rv_metric(resampled, M)
        return minrv
    else:    
        # Use the subsampling utility
        minrv, minrv_ss = calculate_subsampled_metric(
            prices, 
            resample_freq, 
            min_rv_metric, 
            n_subsamples, 
            resampling_method
        )

        return minrv, minrv_ss

# Kernel functions for realized kernel variance
def kernel_type(x, type = 'parzen'):
    # kernel evaluation
    if type == 'parzen':
        if 0 <= x <= 0.5:
            return 1 - 6*x**2 + 6*x**3
        elif 0.5 < x <= 1:
            return 2 * (1 - x)**3
        elif x > 1:
            return 0 
    elif type == 'bartlett':
        return 1 - x if x <= 1 else 0
    elif type == 'tukeyhan':
        return 0.5 * (1 + np.cos(np.pi * x)) if x <= 1 else 0

def resample_with_time_offsets(df, price_col, interval, offset_unit='s', num_offsets=1200):
    """Resample time series with progressive time-based offsets that shift the entire dataframe."""
    RV_sparse_values = []
    
    # Make sure the dataframe has a proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Define the time delta for each offset
    if offset_unit == 's':
        delta = pd.Timedelta(seconds=1)
    elif offset_unit in ['min', 'T']:
        delta = pd.Timedelta(minutes=1)
    else:
        raise ValueError("offset_unit must be 's' for seconds or 'min'/'T' for minutes")
    
    # Get the original start time of the interval
    original_start = df.index[0]
    
    for i in range(num_offsets):
        # Calculate the time offset
        time_offset = i * delta
        
        # Create a new resampling origin that's shifted by the offset
        resample_origin = original_start + time_offset
            
        # Resample using the offset origin
        resampler = df[price_col].resample(
            interval, 
            origin=resample_origin,
            label='right',     # Label uses right bin edge
            closed='right'     # Interval includes right edge
        )
        shifted_prices = resampler.ffill()
        
        # Calculate returns and RV
        shifted_returns = np.diff(np.log(shifted_prices.values))
        
        RV = np.sum(shifted_returns**2)
        RV_sparse_values.append(RV)
    
    # Calculate average RV
    RV_sparse = np.mean(RV_sparse_values)
    return RV_sparse

def omega_squared(df, price_col, sample_freq ='2min'):
    """Estimate ω² using rv at 2 minutes frequency divided by 2n where n are the number of returns different from zero"""
    # Make sure the dataframe has a proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {e}")
    
    # Ensure data is sorted by time
    df = df.sort_index()

    # Extract price series
    prices = df[price_col]

    resampled, _ = resample_prices(prices, sample_freq, 'last') 
    # Calculate standard realized variance
    log_returns = np.diff(np.log(resampled.values))
    rv = (log_returns ** 2).sum()
    zero_threshold = 1e-12  # UHF industry standard
    n_nonzero = (np.abs(log_returns) > zero_threshold).sum()    
    # Calculate ω²
    if n_nonzero > 0:
        omega_squared = rv / (2 * n_nonzero)
    else:
        omega_squared = 0  

    return omega_squared

def estimate_bandwidth(df, m, price_col ='price'):
    rv_sparse = resample_with_time_offsets(df, price_col, '20min', offset_unit='s', num_offsets=1200)
    
    # Noise variance
    noise_variance = omega_squared(df, price_col)

    c_star = ((12)**2/0.269)**(1/5)  # Constant for Parzen kernel 3.5134
        
    # Compute xi squared noiseVariance/IV
    xi_squared = (noise_variance) / (rv_sparse)

    if xi_squared>1:
       xi_squared = 1
  
    # Bandwidth calculation
    # m number of resampled observation used for the kernel estimation
    H = c_star * (xi_squared)**(2/5) * m**(3/5)
    
    return H

def realized_kernel_variance(df, resample_freq='1s', price_col='price', resampling_method='last'):
    """Calculate the realized kernel variance estimate."""
    # Ensure data is sorted by time
    df = df.sort_index()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, format='%H:%M:%S.%f')
        except:
            df.index = pd.to_datetime(df.index)
   
    # End-point handling (local averaging) - "jittering" 
    # Create local averages at the endpoints
    start_avg = df[price_col].iloc[:2].mean()
    end_avg = df[price_col].iloc[-2:].mean()
    
    # Create modified price series with local averages
    mod_prices = df[price_col].copy()
    mod_prices.iloc[0] = start_avg
    mod_prices.iloc[-1] = end_avg

    # Resample prices and compute returns
    resampled, _ = resample_prices(mod_prices, resample_freq, resampling_method)  
  
    resampled_returns = np.diff(np.log(resampled.values))

    n = len(resampled_returns)
    if n <= 1:
        return 0  # Not enough data    
    
    # Estimate bandwidth
    H = estimate_bandwidth(df, n+1, price_col)  
       
    # Calculate the realized kernel
    realized_var = 0

    # For non-integer H, define the set of lags
    if isinstance(H, float) and not H.is_integer():
        # Create lag sequence with the same number of positive and negative values
        H_int = int(np.floor(H))   
    else:
        # For integer H, use the standard range
        H = int(H)
 
    def autocovariance(h):
        """Calculate the autocovariance for a given lag h"""
        return np.sum(resampled_returns[h:] * resampled_returns[:-h]) if h > 0 else np.sum(resampled_returns * resampled_returns)

    # Loop through the lags
    for h in range(-H_int, H_int+1):    
        # Calculate kernel weight
        kernel_weight = kernel_type(abs(h) / (H + 1))
        
        gamma_h= autocovariance(abs(h))

        # Add to realized variance
        realized_var +=  kernel_weight * gamma_h
    
    return realized_var

def replace_nan(df):
    """Replace NaN values in the price column with local averages."""
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
    """Load and prepare tick data from file."""
    # Determine column types and names based on asset type
    if config['asset_type'] in ['forex', 'futures']:
        data_type = {0: 'str', 1: 'float', 2: 'float', 3: 'float', 4: 'int', 5: 'int'}
        data_columns = ['time', 'price', 'bid', 'ask', 'volume', 'trades']
    else:
        data_type = {0: 'str', 1: 'float', 2: 'int', 3: 'int', 4: 'float'}
        data_columns = ['time', 'price', 'volume', 'trades', 'is_not_outlier']

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
        # if 'is_not_outlier' = 0 : overnight, if 'is_not_outlier' = NaN : outliers, if 'is_not_outlier' = 1 : good price
        df = df[(df['is_not_outlier'] != 0) | df['is_not_outlier'].isna()]
        df['price'] = df['price'] * df['is_not_outlier']
        # Replace NaN values with local averages
        df = replace_nan(df)
    
    # Remove any remaining NaN values
    df = df.dropna(subset=['price'])
    
    return df

# ============================================================
# S3 DATA PROCESSOR
# ============================================================

class S3DataProcessor:
    """Handles reading processed data from S3 and writing results to PostgreSQL"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('rv_processor')
        
        # Initialize S3 client
        self.s3_client = self._setup_s3_client()
        
        # Initialize database connection
        self.db_config = self._setup_db_config()
        
    def _setup_s3_client(self):
        """Setup S3 client with credentials from Windmill or environment"""
        try:
            if WINDMILL_AVAILABLE:
                # Get credentials from Windmill
                s3_endpoint_url = wmill.get_variable("u/niccolosalvini27/S3_ENDPOINT_URL")
                s3_access_key = wmill.get_variable("u/niccolosalvini27/S3_ACCESS_KEY")
                s3_secret_key = wmill.get_variable("u/niccolosalvini27/S3_SECRET_KEY")
            else:
                # Fallback to environment variables
                s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://volare.unime.it:9000')
                s3_access_key = os.getenv('S3_ACCESS_KEY', 'minio-forvarduser')
                s3_secret_key = os.getenv('S3_SECRET_KEY', 'RejcApaRjR8k8j3a8MJG')

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
                    'host': wmill.get_variable("u/niccolosalvini27/DB_HOST") or 'forvard_app_postgres',
                    'port': int(wmill.get_variable("u/niccolosalvini27/DB_PORT") or 5432),
                    'database': wmill.get_variable("u/niccolosalvini27/DB_NAME") or 'forvarddb',
                    'user': wmill.get_variable("u/niccolosalvini27/DB_USER") or 'forvarduser',
                    'password': wmill.get_variable("u/niccolosalvini27/DB_PASSWORD") or 'WsUpwXjEA7HHidmL8epF'
                }
            
            self.logger.info(f"Database config: {db_config['host']}:{db_config['port']}/{db_config['database']}")
            return db_config
            
        except Exception as e:
            self.logger.error(f"Failed to setup database config: {e}")
            raise

    def list_s3_files(self, asset_type: str, symbol: str) -> List[str]:
        """List all files for a symbol in S3"""
        try:
            bucket_name = self.config['s3_bucket']
            # Use consistent path structure: data/{asset_type}/{symbol}/
            prefix = f"data/{asset_type}/{symbol}/"
            
            self.logger.info(f"Searching S3 path: {bucket_name}/{prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    
                    # Filter out unwanted files
                    if filename.endswith('.txt') or filename.endswith('.parquet'):
                        if not filename.endswith('_last_update.txt') and filename != 'adjustment.txt':
                            files.append(key)
            
            self.logger.info(f"Found {len(files)} files for {symbol}")
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"Error listing S3 files for {symbol}: {e}")
            return []

    def read_s3_file(self, s3_key: str) -> Optional[pd.DataFrame]:
        """Read a single file from S3"""
        try:
            bucket_name = self.config['s3_bucket']
            
            # Download file content
            response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            file_content = response['Body'].read()
            
            # Determine file format
            file_format = self.config.get('file_format', 'txt').lower()
            
            if file_format == 'parquet':
                # Handle parquet files
                try:
                    import pyarrow.parquet as pq
                    import io
                    df = pq.read_table(io.BytesIO(file_content)).to_pandas()
                except ImportError:
                    self.logger.warning(f"PyArrow not available, attempting CSV read for {s3_key}")
                    # Fallback to CSV reading
                    import io
                    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), header=None)
            else:
                # Handle CSV/TXT files
                import io
                df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), header=None)
            
            self.logger.debug(f"Successfully read {len(df)} rows from {s3_key}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading S3 file {s3_key}: {e}")
            return None

    def save_to_database(self, results: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Save RV results to PostgreSQL database"""
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
        
        try:
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            skipped_count = 0
            inserted_count = 0
            
            for result in results:
                try:
                    # Convert numpy types to Python native types
                    converted_result = {k: convert_numpy_types(v) for k, v in result.items()}
                    
                    # Check if record already exists
                    check_query = """
                        SELECT id FROM realized_volatility_data 
                        WHERE observation_date = %s AND symbol = %s
                    """
                    cursor.execute(check_query, (converted_result['date'], converted_result['symbol']))
                    existing = cursor.fetchone()
                    
                    if existing:
                        skipped_count += 1
                        continue
                    
                    # Insert new record
                    insert_query = """
                        INSERT INTO realized_volatility_data (
                            observation_date, symbol, asset_type, volume, trades,
                            open_price, close_price, high_price, low_price,
                            pv, gk, rr5, rv1, rv5, rv5_ss, bv1, bv5, bv5_ss,
                            rsp1, rsn1, rsp5, rsn5, rsp5_ss, rsn5_ss,
                            medrv1, medrv5, medrv5_ss, minrv1, minrv5, minrv5_ss,
                            rk, rq1, rq5, rq5_ss
                        ) VALUES (
                            %(date)s, %(symbol)s, %(asset_type)s, %(volume)s, %(trades)s,
                            %(open)s, %(close)s, %(high)s, %(low)s,
                            %(pv)s, %(gk)s, %(rr5)s, %(rv1)s, %(rv5)s, %(rv5_ss)s,
                            %(bv1)s, %(bv5)s, %(bv5_ss)s, %(rsp1)s, %(rsn1)s,
                            %(rsp5)s, %(rsn5)s, %(rsp5_ss)s, %(rsn5_ss)s,
                            %(medrv1)s, %(medrv5)s, %(medrv5_ss)s,
                            %(minrv1)s, %(minrv5)s, %(minrv5_ss)s,
                            %(rk)s, %(rq1)s, %(rq5)s, %(rq5_ss)s
                        )
                    """
                    
                    cursor.execute(insert_query, converted_result)
                    inserted_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error inserting record for {result.get('symbol', 'unknown')} {result.get('date', 'unknown')}: {e}")
                    continue
            
            # Commit changes
            conn.commit()
            
            self.logger.info(f"Database save completed: {inserted_count} inserted, {skipped_count} skipped")
            return skipped_count, inserted_count
            
        except Exception as e:
            self.logger.error(f"Database error: {e}")
            if 'conn' in locals():
                conn.rollback()
            return 0, 0
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

# ============================================================
# CORE PROCESSING FUNCTIONS
# ============================================================

def validate_result_data(result: Dict[str, Any]) -> bool:
    """Validate result data before processing"""
    if not isinstance(result, dict):
        return False
    
    required_fields = ['date', 'symbol', 'asset_type']
    for field in required_fields:
        if field not in result or result[field] is None or result[field] == '':
            return False
    
    try:
        datetime.strptime(result['date'], '%Y-%m-%d')
    except ValueError:
        return False
    
    # Check for infinite or NaN values in numeric fields
    numeric_fields = ['volume', 'trades', 'open', 'close', 'high', 'low', 'pv', 'gk', 'rr5', 
                     'rv1', 'rv5', 'rv5_ss', 'bv1', 'bv5', 'bv5_ss', 'rsp1', 'rsn1', 
                     'rsp5', 'rsn5', 'rsp5_ss', 'rsn5_ss', 'medrv1', 'medrv5', 'medrv5_ss',
                     'minrv1', 'minrv5', 'minrv5_ss', 'rk', 'rq1', 'rq5', 'rq5_ss']
    
    for field in numeric_fields:
        if field in result and result[field] is not None:
            if np.isnan(result[field]) or np.isinf(result[field]):
                return False
    
    return True

def process_single_file_with_retry(s3_processor: S3DataProcessor, s3_key: str, 
                                 file_date: datetime, symbol: str, config: Dict[str, Any], 
                                 max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """Process a single S3 data file with retry mechanism - IDENTICAL LOGIC TO ORIGINAL"""
    logger = logging.getLogger('rv_processor')
    
    for attempt in range(max_retries + 1):
        try:
            # Read file from S3
            raw_data = s3_processor.read_s3_file(s3_key)
            if raw_data is None or raw_data.empty:
                return None
            
            # Prepare data using IDENTICAL prepare_data_rv function
            # First need to save to temp file for prepare_data_rv function
            import tempfile
            import os
            
            # Determine file format and extension
            file_format = config.get('file_format', 'txt').lower()
            file_extension = '.parquet' if file_format == 'parquet' else '.txt'
            
            with tempfile.NamedTemporaryFile(mode='w+b', suffix=file_extension, delete=False) as temp_file:
                temp_file_path = temp_file.name
                
                if file_format == 'parquet':
                    raw_data.to_parquet(temp_file_path, index=False)
                else:
                    raw_data.to_csv(temp_file_path, index=False, header=False)
            
            try:
                # Use IDENTICAL prepare_data_rv function
                df = prepare_data_rv(temp_file_path, config)
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
                
            if df is None or df.empty:
                return None
           
            # Extract OHLC values - IDENTICAL LOGIC
            if config['asset_type'] in ['futures', 'forex']:
                num_trades = df['trades'].sum() if 'trades' in df.columns else 0
                volume = 0
            else:
                num_trades = df['trades'].sum() if 'trades' in df.columns else 0
                volume = df['volume'].sum() if 'volume' in df.columns else 0

            # Calculate volatility measures with error handling - IDENTICAL LOGIC
            volatility_results = {}
            
            try:
                volatility_results['rv1'] = realized_power_variation(df, exp = 2, resample_freq='1min', price_col='price', 
                              resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                rv5_result = realized_power_variation(df, exp = 2, resample_freq='5min', price_col='price', 
                              resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['rv5'] = rv5_result[0] if isinstance(rv5_result, tuple) else rv5_result
                volatility_results['rv5_ss'] = rv5_result[1] if isinstance(rv5_result, tuple) else None
            except Exception:
                volatility_results.update({'rv1': None, 'rv5': None, 'rv5_ss': None})
                
            # Quarticity
            try:
                volatility_results['rq1'] = realized_power_variation(df, exp= 4, resample_freq='1min', price_col='price', 
                              resampling_method='last', calculate_subsample=False, n_subsamples=5)
           
                rq5_result = realized_power_variation(df, exp=4, resample_freq='5min', price_col='price', 
                              resampling_method='last', calculate_subsample=True, n_subsamples=5)
                
                volatility_results['rq5'] = rq5_result[0] if isinstance(rq5_result, tuple) else rq5_result
                volatility_results['rq5_ss'] = rq5_result[1] if isinstance(rq5_result, tuple) else None
               
            except Exception:
                volatility_results.update({'rq1': None, 'rq5': None, 'rq5_ss': None})
            
            try:
                volatility_results['bv1'] = bipower_variation(df, resample_freq='1min', price_col='price', 
                              resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                bv5_result = bipower_variation(df, resample_freq='5min', price_col='price', 
                              resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['bv5'] = bv5_result[0] if isinstance(bv5_result, tuple) else bv5_result
                volatility_results['bv5_ss'] = bv5_result[1] if isinstance(bv5_result, tuple) else None
            except Exception:
                volatility_results.update({'bv1': None, 'bv5': None, 'bv5_ss': None})
            
            try:
                rs_result_1 = realized_semivariance(df, resample_freq='1min', price_col='price', 
                                 resampling_method='last', calculate_subsample=False, n_subsamples=5)
                volatility_results['rsp1'] = rs_result_1[0] if isinstance(rs_result_1, tuple) else None
                volatility_results['rsn1'] = rs_result_1[1] if isinstance(rs_result_1, tuple) else None
                
                rs_result_5 = realized_semivariance(df, resample_freq='5min', price_col='price', 
                                 resampling_method='last', calculate_subsample=True, n_subsamples=5)
                if isinstance(rs_result_5, tuple) and len(rs_result_5) >= 4:
                    volatility_results['rsp5'] = rs_result_5[0]
                    volatility_results['rsn5'] = rs_result_5[1]
                    volatility_results['rsp5_ss'] = rs_result_5[2]
                    volatility_results['rsn5_ss'] = rs_result_5[3]
                else:
                    volatility_results.update({'rsp5': None, 'rsn5': None, 'rsp5_ss': None, 'rsn5_ss': None})
            except Exception:
                volatility_results.update({'rsp1': None, 'rsn1': None, 'rsp5': None, 'rsn5': None, 
                                         'rsp5_ss': None, 'rsn5_ss': None})
            
            try:
                volatility_results['medrv1'] = median_realized_variance(df, resample_freq='1min', price_col='price', 
                                    resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                medrv5_result = median_realized_variance(df, resample_freq='5min', price_col='price', 
                                    resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['medrv5'] = medrv5_result[0] if isinstance(medrv5_result, tuple) else medrv5_result
                volatility_results['medrv5_ss'] = medrv5_result[1] if isinstance(medrv5_result, tuple) else None
            except Exception:
                volatility_results.update({'medrv1': None, 'medrv5': None, 'medrv5_ss': None})
            
            try:
                volatility_results['minrv1'] = min_realized_variance(df, resample_freq='1min', price_col='price', 
                                    resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                minrv5_result = min_realized_variance(df, resample_freq='5min', price_col='price', 
                                    resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['minrv5'] = minrv5_result[0] if isinstance(minrv5_result, tuple) else minrv5_result
                volatility_results['minrv5_ss'] = minrv5_result[1] if isinstance(minrv5_result, tuple) else None
            except Exception:
                volatility_results.update({'minrv1': None, 'minrv5': None, 'minrv5_ss': None})

            try:
                volatility_results['rk'] = realized_kernel_variance(df, resample_freq='1s', price_col='price', resampling_method='last')
            except Exception:
                volatility_results['rk'] = None
            
            try:
                high_price, low_price, pv = parkinson_variance(df)
                rr5 = realized_range(df)
            except Exception:
                high_price = low_price = pv = rr5 = None
            
            try:
                open_price, close_price, gk = garman_klass_variance(df, config['asset_type'])
            except Exception:
                open_price = close_price = gk = None
           
            result = {
                'date': file_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'asset_type': config['asset_type'],
                'volume': volume,
                'trades': num_trades,
                'open': open_price,
                'close': close_price,
                'high': high_price,
                'low': low_price,
                'pv': pv,
                'gk': gk,
                'rr5': rr5,
                **volatility_results
            }
            
            if validate_result_data(result):
                return result
            else:
                return None
        
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed for {s3_key}: {e}. Retrying...")
                time.sleep(1)
            else:
                logger.error(f"All attempts failed for {s3_key}: {e}")
                return None

def process_symbol_files(symbol: str, config: Dict[str, Any], s3_processor: S3DataProcessor) -> Dict[str, Any]:
    """Process all files for a single symbol"""
    logger = logging.getLogger('rv_processor')
    start_time = time.time()
    
    try:
        asset_type = config['asset_type']
        
        # List files from S3
        s3_files = s3_processor.list_s3_files(asset_type, symbol)
        
        if not s3_files:
            logger.warning(f"No files found for {symbol}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "total_files": 0}
        
        # Extract valid date files
        valid_files = []
        date_pattern = r'(\d{4}_\d{2}_\d{2})'
        
        for s3_key in s3_files:
            filename = s3_key.split('/')[-1]
            date_match = re.search(date_pattern, filename)
            if date_match:
                valid_files.append((s3_key, date_match))
        
        if not valid_files:
            logger.warning(f"No valid date-formatted files for {symbol}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "total_files": 0}
        
        total_files = len(valid_files)
        logger.info(f"Processing {symbol}: {total_files} files")
        
        # Process files in batches
        batch_size = min(20, max(5, len(valid_files) // 10))
        file_batches = [valid_files[i:i+batch_size] for i in range(0, len(valid_files), batch_size)]
        
        processed = 0
        errors = 0
        
        for batch in file_batches:
            results_batch = []
            
            for s3_key, date_match in batch:
                try:
                    file_date = datetime.strptime(date_match.group(1), '%Y_%m_%d')
                    result = process_single_file_with_retry(s3_processor, s3_key, file_date, symbol, config)
                    
                    if result:
                        results_batch.append(result)
                    else:
                        errors += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {s3_key}: {e}")
                    errors += 1
            
            # Save batch to database
            if results_batch:
                try:
                    skipped, batch_processed = s3_processor.save_to_database(results_batch)
                    processed += batch_processed
                except Exception as e:
                    logger.error(f"Error saving batch for {symbol}: {e}")
                    errors += len(results_batch)
        
        processing_time = time.time() - start_time
        
        logger.info(f"{symbol} completed in {processing_time:.1f}s: {processed} processed, {errors} errors")
        
        return {
            "symbol": symbol, 
            "processed": processed, 
            "skipped": 0,  # Database handles duplicates
            "errors": errors, 
            "total_files": total_files,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Critical error processing {symbol}: {e}")
        return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, 
                "total_files": 0, "processing_time": 0}

def process_all_symbols_threaded(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process all symbols using thread-based parallelism - IDENTICAL TO ORIGINAL STRUCTURE"""
    logger = logging.getLogger('rv_processor')
    
    if not config.get('symbols'):
        logger.error("No symbols provided")
        return []
    
    # Initialize S3 processor
    s3_processor = S3DataProcessor(config)
    
    num_symbols = len(config['symbols'])
    max_workers = config.get('max_workers', min(max(1, os.cpu_count() - 1), num_symbols, 4))
    
    logger.info("Starting realized variance processing")
    logger.info(f"Workers: {max_workers}, Symbols: {num_symbols}, Asset type: {config['asset_type']}")
    
    start_time = time.time()
    results = []
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(process_symbol_files, symbol, config, s3_processor): symbol 
                for symbol in config['symbols']
            }
            
            completed = 0
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=3600)
                    results.append(result)
                    completed += 1
                    
                    progress = (completed / num_symbols) * 100
                    status = "SUCCESS" if result['errors'] == 0 else "WARNING"
                    logger.info(f"Progress: {completed}/{num_symbols} ({progress:.0f}%) - "
                            f"COMPLETED: {symbol} ({status}) - "
                            f"{result['processed']} processed, "
                            f"{result['errors']} errors in {result.get('processing_time', 0):.1f}s")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results.append({
                        "symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, 
                        "total_files": 0, "processing_time": 0
                    })
                    completed += 1
    
    except Exception as e:
        logger.error(f"Critical error in thread execution: {e}")
        return results
    
    # Summary
    total_elapsed = time.time() - start_time
    total_processed = sum(r["processed"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_files = sum(r.get("total_files", 0) for r in results)
    
    successful_symbols = [r for r in results if r['errors'] == 0]
    symbols_with_errors = [r for r in results if r["errors"] > 0]
    
    logger.info("Processing completed")
    logger.info(f"Total time: {total_elapsed:.1f}s, Files: {total_files}, Processed: {total_processed}, Errors: {total_errors}")
    logger.info(f"Symbols: {len(results)} total, {len(successful_symbols)} successful, {len(symbols_with_errors)} with errors")
    
    if symbols_with_errors:
        logger.warning("Symbols with errors:")
        for r in symbols_with_errors:
            logger.warning(f"  {r['symbol']}: {r['errors']} errors")
    
    return results

# ============================================================
# CONFIGURATION AND MAIN EXECUTION
# ============================================================

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger('rv_processor')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger



def main(
    # Pipeline configuration
    pipeline_name="realized_volatility",
    pipeline_enabled=True,
    
    # Asset type configuration - TRUE/FALSE per ogni tipo
    stocks_enabled=True,
    stocks_symbols=["MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY", "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA", "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM", "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK", "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC"],
    forex_enabled=True,
    forex_symbols=[  "EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "JPYUSD", "CHFUSD", "SGDUSD", "HKDUSD", "KRWUSD", "INRUSD", "RUBUSD", "BRLUSD"],
    futures_enabled=True,
    futures_symbols=[  "ES", "CL", "GC", "NG", "NQ", "TY", "FV", "EU", "SI", "C", "W", "VX"],
    
    # MANCANO GLI ETF

    # Processing settings
    max_workers=4,
    file_format="parquet",
    s3_bucket=None
):
    """
    Standalone Realized Variance Computation
    
    Args:
        pipeline_name: Name of the pipeline (e.g., 'stocks_batch1', 'forex', 'futures')
        pipeline_enabled: Whether the pipeline is enabled
        
        # Asset type configuration
        stocks_enabled: Enable stocks pipeline
        stocks_symbols: List of stock symbols
        forex_enabled: Enable forex pipeline  
        forex_symbols: List of forex symbols
        futures_enabled: Enable futures pipeline
        futures_symbols: List of futures symbols
        
        max_workers: Maximum number of parallel workers
        file_format: File format ('parquet' or 'txt')
        s3_bucket: S3 bucket name (if None, will use Windmill variable)
    """
    start_time = datetime.now()
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"REALIZED VARIANCE COMPUTATION START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Get S3 bucket from Windmill if not provided
        if s3_bucket is None:
            s3_bucket = wmill.get_variable("u/niccolosalvini27/S3_BUCKET")
        
        # Log which asset types are enabled
        enabled_types = []
        if stocks_enabled:
            enabled_types.append(f"stocks({len(stocks_symbols)} symbols)")
        if forex_enabled:
            enabled_types.append(f"forex({len(forex_symbols)} symbols)")
        if futures_enabled:
            enabled_types.append(f"futures({len(futures_symbols)} symbols)")
        
        if enabled_types:
            logger.info(f"Enabled asset types: {', '.join(enabled_types)}")
        else:
            logger.info("No asset types enabled, using default (stocks)")
            stocks_enabled = True
            stocks_symbols = ["GE", "JNJ"]
        
        # Process all enabled asset types
        all_results = []
        total_processed = 0
        total_errors = 0
        total_files = 0
        total_skipped = 0
        
        # Process stocks if enabled
        if stocks_enabled:
            logger.info("Processing STOCKS asset type...")
            stocks_config = {
                'asset_type': "stocks",
                'symbols': stocks_symbols,
                'file_format': file_format,
                'max_workers': max_workers,
                's3_bucket': s3_bucket
            }
            stocks_results = process_all_symbols_threaded(stocks_config)
            all_results.extend(stocks_results)
            total_processed += sum(r["processed"] for r in stocks_results)
            total_errors += sum(r["errors"] for r in stocks_results)
            total_files += sum(r.get("total_files", 0) for r in stocks_results)
            total_skipped += sum(r.get("skipped", 0) for r in stocks_results)
        
        # Process forex if enabled
        if forex_enabled:
            logger.info("Processing FOREX asset type...")
            forex_config = {
                'asset_type': "forex",
                'symbols': forex_symbols,
                'file_format': file_format,
                'max_workers': max_workers,
                's3_bucket': s3_bucket
            }
            forex_results = process_all_symbols_threaded(forex_config)
            all_results.extend(forex_results)
            total_processed += sum(r["processed"] for r in forex_results)
            total_errors += sum(r["errors"] for r in forex_results)
            total_files += sum(r.get("total_files", 0) for r in forex_results)
            total_skipped += sum(r.get("skipped", 0) for r in forex_results)
        
        # Process futures if enabled
        if futures_enabled:
            logger.info("Processing FUTURES asset type...")
            futures_config = {
                'asset_type': "futures",
                'symbols': futures_symbols,
                'file_format': file_format,
                'max_workers': max_workers,
                's3_bucket': s3_bucket
            }
            futures_results = process_all_symbols_threaded(futures_config)
            all_results.extend(futures_results)
            total_processed += sum(r["processed"] for r in futures_results)
            total_errors += sum(r["errors"] for r in futures_results)
            total_files += sum(r.get("total_files", 0) for r in futures_results)
            total_skipped += sum(r.get("skipped", 0) for r in futures_results)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        successful_symbols = [r for r in all_results if r['errors'] == 0]
        success_rate = len(successful_symbols) / len(all_results) * 100 if all_results else 0
        
        logger.info("FINAL SUMMARY")
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Pipeline: {pipeline_name}")
        logger.info(f"Total processed: {total_processed}, Errors: {total_errors}, Success rate: {success_rate:.0f}%")
        
        # Determine status
        if total_errors == 0:
            logger.info("ALL PROCESSING COMPLETED SUCCESSFULLY!")
            status_emoji = "✅"
            status_text = "SUCCESS"
        else:
            logger.warning("SOME PROCESSING ERRORS OCCURRED!")
            status_emoji = "⚠️" if total_errors < total_processed else "❌"
            status_text = "PARTIAL FAILURE" if total_processed > 0 else "FAILURE"
        
        # Create Slack message
        slack_message = f"{status_emoji} *REALIZED VARIANCE COMPUTATION COMPLETED* {status_emoji}\n\n"
        slack_message += f"*Status:* {status_text}\n"
        slack_message += f"*Pipeline:* {pipeline_name}\n"
        slack_message += f"*Duration:* {str(duration).split('.')[0]}\n"
        slack_message += f"*Total Symbols:* {len(all_results)}\n\n"
        
        # Add enabled asset types info
        if enabled_types:
            slack_message += f"*Enabled Types:* {', '.join(enabled_types)}\n\n"
        
        slack_message += "*Processing Summary:*"
        slack_message += f"\n📁 Total files: {total_files}"
        slack_message += f"\n✅ Records processed: {total_processed}"
        slack_message += f"\n⏭️ Records skipped: {total_skipped}"
        slack_message += f"\n❌ Processing errors: {total_errors}"
        slack_message += f"\n🏷️ Symbols processed: {len(all_results)}"
        
        if total_files > 0:
            slack_message += f"\n📊 Success rate: {success_rate:.1f}%"
        
        slack_message += f"\n\n*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send Slack notification
        try:
            send_slack_notification(slack_message)
        except Exception as e:
            logger.warning(f"Error sending Slack notification: {e}")
        
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")