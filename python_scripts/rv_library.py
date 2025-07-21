"""
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics. 
Research code - Unauthorized distribution prohibited.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import numpy as np


# ============================================================
# REALIZED VARIANCE MODELS
# ============================================================

def parkinson_variance(data):
    """
    Calculate Parkinson's Range-based Volatility Estimator for High-Frequency Data 
    """
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
    """
    Calculate Garman-Klass Range-based Volatility Estimator for High-Frequency Data  
    """
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
    """
    Calculate the realized range proposed by Martens and Van Dijk (2007) and 
    Christensen and Podolskij (2007)
    
    """
    prices = (data['price'])
        
    # Resample the data to the desired frequency
    # We need to delete the first observation which set instead of nan the first value of the series
    high = np.log(prices.resample('5T', label='right', closed='right').max())[1:]
    low = np.log(prices.resample('5T', label='right', closed='right').min())[1:]
  
    # Calculate the realized variance range for each period
    scaling_factor = 1/( 4 * np.log(2) )# Scale by the theoretical constant (4 log 2)
    realized_range = scaling_factor * np.sum((high - low) ** 2) 
    
    return realized_range

'''def resample_prices(prices, resample_freq='5T', resampling_method='last'):
    if resampling_method == 'mean':
        resampled = prices.resample(resample_freq, label='right', closed='right').mean()
    elif resampling_method == 'last':
        resampled = prices.resample(resample_freq, label='right', closed='right').last()
    elif resampling_method == 'median':
        resampled = prices.resample(resample_freq, label='right', closed='right').median()
    else:
        raise ValueError(f"Method '{resampling_method}' not supported.")
    first_available_price = prices.iloc[0]  # Get the first available price
    first_time = prices.index[0].replace(second=0, microsecond=0)  # Set the first time to 9:30:00 or equivalent
    resampled.loc[first_time] = first_available_price  # Add the first value to the resampled data
    resampled = resampled.sort_index().ffill()
    #print(resampled)
    return resampled'''

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

def calculate_subsampled_metric(prices, resample_freq, metric_func, 
                               n_subsamples=5, resampling_method='last'):
    """
    Calculate subsampled metric with consistent backfill logic.
    """
    #print('prices', prices)
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
                    
            #print(resampled_i)
                        
        except Exception as e:
            print(f"Error processing subsample {i}: {e}")
            continue

    
    if len(subsample_metrics) == 0:
        return np.nan, np.nan
    
    # First metric is standard, average is subsampled
    standard_metric = subsample_metrics[0]
    subsampled_metric = np.mean(subsample_metrics)
    
    return standard_metric, subsampled_metric


def realized_power_variation(data, exp=2, resample_freq='5T', price_col='price', 
                      resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """
    Calculate Realized Variance (exp=2) or Realized Quarticity (exp=4) with subsampling.
    """
    
    def realized_power_variation_metric(resampled_prices, M, exp=2):
        """Calculate realized power variation from resampled prices."""
        log_returns = np.log(resampled_prices).diff().dropna()
        rv = (log_returns ** exp).sum()
        
        if exp == 4:
            #print(f'len(log_returns): {len(log_returns)}, M: {M}')
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

   
def bipower_variation(data, resample_freq='5T', price_col='price', 
                      resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """
    Calculate Bipower Variation with subsampling.
    """

    def bipower_variation_metric(resampled_prices, M = None):
        """Calculate bipower variation from resampled prices."""
        log_returns = np.log(resampled_prices).diff().dropna()
        lagged_returns = log_returns.shift(1)
        return_products = np.abs(log_returns) * np.abs(lagged_returns)
        return_products = return_products.dropna()
        
        bv = (np.pi / 2) * np.sum(return_products)
        
        # Debiased measure
        #m = len(log_returns)
        #m = (np.abs(log_returns) > 1e-12).sum()
        #bv_debiased = m/(m - 1) * bv if m > 1 else bv
        
        return bv #bv_debiased

    
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
      
def realized_semivariance(data, resample_freq='5T', price_col='price', 
                         resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """
    Calculate Realized Semi-variance (RSV) and optionally Subsampled RSV (RSV_SS)
    Separates the realized variance into positive and negative components.
    
    """
    def realized_positive_semivariance_metric(resampled_prices, M=None):
        """
        Calculate realized semivariance (positive and negative) from resampled prices.
        
        """
        # Calculate log returns
        log_returns = np.log(resampled_prices).diff().dropna()
        #log_returns = np.diff(np.log(resampled_prices.values))
        
        rsv_positive = (log_returns[log_returns > 0] ** 2).sum()
        
        return rsv_positive
    
    def realized_negative_semivariance_metric(resampled_prices, M=None):
        """
        Calculate realized semivariance (positive and negative) from resampled prices.
        
        """
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

def median_realized_variance(data, resample_freq='5T', price_col='price', 
                            resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """
    Calculate Median Realized Variance (MedRV) with subsampling.
    This estimator is more robust to jumps and microstructure noise.
    """
    def median_rv_metric(resampled_prices, M=None):
        """Calculate median realized variance from resampled prices."""
        # Calculate log returns
        #log_price = np.log(resampled_prices)
        #log_returns = log_price.diff().dropna()
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
        medRV = medRV * medRVScale * (M/(M-2))#* (n/(n-2))
          
        
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
        

def min_realized_variance(data, resample_freq='5T', price_col='price', 
                         resampling_method='last', calculate_subsample=True, n_subsamples=5):
    """
    Calculate Minimum Realized Variance (MinRV) with subsampling.
    """
    def min_rv_metric(resampled_prices, M=None):
        """Calculate minimum realized variance from resampled prices."""
        # Calculate log returns
        #log_price = np.log(resampled_prices)
        #log_returns = log_price.diff().dropna()
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
        minRV = minRV * minRVscale * (M/(M-2))#* (n/(n-1))
        
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
   
    

# Kernel (12_05_2025)
# OK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def resample_with_time_offsets(df, price_col, interval, offset_unit='S', num_offsets=1200):
    """
    Resample time series with progressive time-based offsets that shift the entire dataframe.
    """
    RV_sparse_values = []
    
    # Make sure the dataframe has a proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Define the time delta for each offset
    if offset_unit == 'S':
        delta = pd.Timedelta(seconds=1)
    elif offset_unit in ['T', 'min']:
        delta = pd.Timedelta(minutes=1)
    else:
        raise ValueError("offset_unit must be 'S' for seconds or 'T'/'min' for minutes")
    
    # Get the original start time of the interval
    original_start = df.index[0]
    
    for i in range(num_offsets):
        # Calculate the time offset
        time_offset = i * delta
        
        # Create a new resampling origin that's shifted by the offset
        resample_origin = original_start + time_offset
            
        # Resample using the offset origin
        # We use 'origin' parameter to control the starting point of the resampling bins
          # Resample using the offset origin with specific parameters
        resampler = df[price_col].resample(
            interval, 
            origin=resample_origin,
            label='right',     # Label uses right bin edge
            closed='right'     # Interval includes right edge
        )
        shifted_prices = resampler.ffill() # da controllare
        #print(shifted_prices)
        
        # Calculate returns and RV
        #shifted_returns = np.log(shifted_prices).diff().dropna()
        shifted_returns = np.diff(np.log(shifted_prices.values)) # <--- usare questo
        
        
        
        RV = np.sum(shifted_returns**2)
        RV_sparse_values.append(RV)
    
    # Calculate average RV
    RV_sparse = np.mean(RV_sparse_values)
    return RV_sparse

def omega_squared(df, price_col, sample_freq ='2T'):
    """
    Estimate ω² using rv at 2 minutes frequency divided by 2n where n are the number of returns different from zero
    """
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
    #n_nonzero = (log_returns != 0).sum()    
    zero_threshold = 1e-12  # UHF industry standard
    n_nonzero = (np.abs(log_returns) > zero_threshold).sum()    
    # Calculate ω²
    if n_nonzero > 0:
        omega_squared = rv / (2 * n_nonzero)
    else:
        omega_squared = 0  

    return omega_squared

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

def estimate_bandwidth(df, m, price_col ='price'):
    
    rv_sparse = resample_with_time_offsets(df, price_col, '20T', offset_unit='S', num_offsets=1200)  
    #print('rv_sparse', rv_sparse)
    
    # Noise variance
    noise_variance = omega_squared(df, price_col)
    #print('noise_variance', noise_variance)

    c_star = ((12)**2/0.269)**(1/5)  # Constant for Parzen kernel 3.5134
        
    # Compute xi squared noiseVariance/IV
    xi_squared = (noise_variance) / (rv_sparse)

    if xi_squared>1:
       #print('IQ estimaed o be close to 0')
       xi_squared = 1
  
    # Bandwidth calculation
    # m number of resampled observation used for the kernel estimation
    H = c_star * (xi_squared)**(2/5) * m**(3/5)
    
    return H #max(min(H, m//4), 2) 

def realized_kernel_variance(df, resample_freq='1S', price_col='price', resampling_method='last'):
    """
        Calculate the realized kernel variance estimate.    
    """
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
  
    #resampled_returns = np.log(resampled).diff().dropna() NON so perchè ma mi da risultati diversi se uso questo per outocov
    resampled_returns = np.diff(np.log(resampled.values))

    n = len(resampled_returns)
    if n <= 1:
        return 0  # Not enough data    
    
    # Estimate bandwidth
    H = estimate_bandwidth(df, n+1, price_col)  
    #print('H:', H)
    #H = 5.6777 Sheppard considering 5T    
       
    
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
    #for h in lags: 
    for h in range(-H_int, H_int+1):    
        # Calculate kernel weight
        kernel_weight = kernel_type(abs(h) / (H + 1))
        
        gamma_h= autocovariance(abs(h))

        # Add to realized variance
        realized_var +=  kernel_weight * gamma_h
    
    return realized_var


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

    # Ensure microseconds are present in time format
    #if isinstance(df['time'].iloc[0], str):
    #    df['time'] = df['time'].apply(lambda t: t + '.000' if '.' not in t else t)

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
    #print(df.isna().sum())
    
    return df

