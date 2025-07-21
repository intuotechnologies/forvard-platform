
"""
Realized Covariance Library
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 01/07/2025

2025 University of Messina, Department of Economics.
Research code - Unauthorized distribution prohibited.
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from numba import jit, prange
from rv_library import prepare_data_rv, resample_prices



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
            #print(asset, len(resampled_prices))
            #print(resampled_prices)
            price_series[asset] = resampled_prices

            
        except Exception as e:
            if logger:
                logger.error(f"Error processing {asset} on {date}: {e}")
            continue
    
    if len(price_series) < 2:
        return None
    
    try:
      
        #sync_prices = pd.DataFrame(price_series).ffill()

        # ====
        # MANTIENE l'ordine dal file di configurazione originale
        original_asset_order = list(asset_configs.keys())  # Ordine dal file symbols
        available_assets = [asset for asset in original_asset_order if asset in price_series]
        ordered_price_series = {asset: price_series[asset] for asset in available_assets}
        sync_prices = pd.DataFrame(ordered_price_series).ffill()
        # ====
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
