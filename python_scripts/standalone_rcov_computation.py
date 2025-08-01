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

def resample_prices(prices, resample_freq='1T', resampling_method='last'):
    """
    Resample price series with specified frequency and method.
    """
    try:
        # Resample with specified frequency and method
        if resampling_method == 'last':
            resampled = prices.resample(resample_freq).last()
        elif resampling_method == 'mean':
            resampled = prices.resample(resample_freq).mean()
        elif resampling_method == 'first':
            resampled = prices.resample(resample_freq).first()
        else:
            resampled = prices.resample(resample_freq).last()
        
        # Forward fill missing values
        resampled = resampled.ffill()
        
        # Drop NaN values
        resampled = resampled.dropna()
        
        return resampled, len(resampled)
    
    except Exception as e:
        logger.error(f"Error in resampling: {e}")
        return prices, len(prices)

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

def prepare_data_rcov(file_path, config, s3_client, bucket_name):
    """Load and prepare tick data from S3 file."""
    # Determine column types and names based on asset type
    if config['asset_type'] in ['forex', 'futures']:
        data_type = {0: 'str', 1: 'float', 2: 'float', 3: 'float', 4: 'int', 5: 'int'}
        data_columns = ['time', 'price', 'bid', 'ask', 'volume', 'trades']
    else:
        data_type = {0: 'str', 1: 'float', 2: 'int', 3: 'int', 4: 'float'}
        data_columns = ['time', 'price', 'volume', 'trades', 'is_not_outlier']

    # Get file format from config, default to 'parquet'
    file_format = config.get('file_format', 'parquet').lower()
   
    try:
        # Read the file based on format from S3
        if file_format == 'parquet':
            # Read parquet file from S3 using BytesIO
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            parquet_data = BytesIO(response['Body'].read())
            df = pd.read_parquet(parquet_data)
            
            # Ensure column names are consistent
            if len(df.columns) == len(data_columns):
                df.columns = data_columns
        else:
            # Read CSV (txt) file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            csv_data = StringIO(response['Body'].read().decode('utf-8'))
            df = pd.read_csv(csv_data, header=None, dtype=data_type)
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
        
        # Sort by time to ensure proper ordering
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def process_single_day(file_paths, assets, date, asset_configs, resample_freq='1T', 
                      resampling_method='last', measures=['RCov'], logger=None,
                      mixed_asset_mode=False, early_closing_day_file=None, s3_client=None, bucket_name=None):
    """
    Process a single day of data for covariance calculation.
    This function maintains the exact same logic as the original rcov_library.py
    """
    if not file_paths or len(file_paths) < 2:
        return None
    
    price_series = {}
    has_stocks = any(config['asset_type'] == 'stocks' for config in asset_configs.values())
    
    # Load and process each asset's data
    for asset in assets:
        if asset not in file_paths:
            continue
            
        asset_config = asset_configs.get(asset, {})
        file_path = file_paths[asset]
        
        try:
            # Load data using the prepare_data_rcov function
            df = prepare_data_rcov(file_path, asset_config, s3_client, bucket_name)
            
            if df is None or df.empty or len(df) < 10:
                continue
            
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
                    continue
            
            # Use price column
            prices = df['price']
            
            # Resample prices
            resampled_prices, _ = resample_prices(prices, resample_freq, resampling_method)
            
            if len(resampled_prices) > 0:
                price_series[asset] = resampled_prices
                
        except Exception as e:
            if logger:
                logger.error(f"Error processing {asset} for {date}: {e}")
            continue
    
    if len(price_series) < 2:
        return None
    
    try:
        # Maintain the order from the original asset configuration
        original_asset_order = list(asset_configs.keys())
        available_assets = [asset for asset in original_asset_order if asset in price_series]
        ordered_price_series = {asset: price_series[asset] for asset in available_assets}
        sync_prices = pd.DataFrame(ordered_price_series).ffill()
        
        # Calculate returns
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
                    'database': wmill.get_variable("u/niccolosalvini27/DB_NAME_DEV") or 'forvarddb_dev',
                    'user': wmill.get_variable("u/niccolosalvini27/DB_USER") or 'forvarduser',
                    'password': wmill.get_variable("u/niccolosalvini27/DB_PASSWORD") or 'WsUpwXjEA7HHidmL8epF'
                }
            else:
                # Fallback to environment variables
                db_config = {
                    'host': os.getenv('DB_HOST', 'forvard_app_postgres'),
                    'port': int(os.getenv('DB_PORT', 5432)),
                    'database': os.getenv('DB_NAME_DEV', 'forvarddb_dev'),
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
        
        for symbol in symbols:
            symbol_config = self.config['symbols'][symbol]
            asset_type = symbol_config['asset_type']
            file_format = symbol_config.get('file_format', 'parquet')
            
            # Construct S3 prefix path
            s3_prefix = f"data/{asset_type}/{symbol}/"
            
            try:
                self.logger.info(f"Searching S3 path: s3://{self.bucket_name}/{s3_prefix}")
                
                # List objects with prefix
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=s3_prefix
                )
                
                if 'Contents' not in response:
                    continue
                
                files = [obj['Key'] for obj in response['Contents']]
                self.logger.info(f"Found {len(files)} files for {symbol}")
                
                # Find file for specific date
                target_filename = f"{symbol}_{date}.{file_format}"
                
                for file_key in files:
                    if file_key.endswith(target_filename):
                        file_paths[symbol] = file_key
                        break
                
            except Exception as e:
                self.logger.error(f"Error searching files for {symbol}: {e}")
                continue
        
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
            resample_freq = self.config['processing'].get('resample_freq', '1min')
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
                result_dict = {
                    'date': row['date'],  # Map 'date' to 'date'
                    'asset1': row['asset1'],   # Map 'asset1' to 'asset1'
                    'asset2': row['asset2'],   # Map 'asset2' to 'asset2'
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
        """Save covariance results to PostgreSQL database"""
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
                        SELECT id FROM realized_covariance_data 
                        WHERE date = %s AND asset1 = %s AND asset2 = %s
                    """
                    cursor.execute(check_query, (
                        converted_result['date'], 
                        converted_result['asset1'], 
                        converted_result['asset2']
                    ))
                    existing = cursor.fetchone()
                    
                    if existing:
                        skipped_count += 1
                        continue
                    
                    # Insert new record
                    insert_query = """
                        INSERT INTO realized_covariance_data (
                            date, asset1, asset2,
                            rcov, rbpcov, rscov_p, rscov_n, rscov_mp, rscov_mn
                        ) VALUES (
                            %(date)s, %(asset1)s, %(asset2)s,
                            %(rcov)s, %(rbpcov)s, %(rscov_p)s, %(rscov_n)s, 
                            %(rscov_mp)s, %(rscov_mn)s
                        )
                    """
                    
                    cursor.execute(insert_query, converted_result)
                    inserted_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error inserting record for {result.get('asset1', 'unknown')}-{result.get('asset2', 'unknown')} {result.get('date', 'unknown')}: {e}")
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
    pipeline_name="futures",
    pipeline_enabled=True,
    
    # Asset type configuration
    stocks_enabled=False,
    stocks_symbols=["GE", "JNJ"],
    forex_enabled=False,
    forex_symbols=["EURUSD", "JPYUSD"],
    futures_enabled=True,
    futures_symbols=["CL", "GC"],
    mixed_enabled=False,
    mixed_symbols=["GE:stocks", "JNJ:stocks", "EURUSD:forex", "JPYUSD:forex", "CL:futures", "GC:futures"],
    
    # Date range
    start_date="03/01/2024",
    end_date="03/01/2025",
    
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
    elif forex_enabled:
        asset_type = "forex"
        symbols_list = forex_symbols
    elif futures_enabled:
        asset_type = "futures"
        symbols_list = futures_symbols
    elif mixed_enabled:
        asset_type = "mixed"
        symbols_list = mixed_symbols
    else:
        # Default to futures if nothing is enabled
        asset_type = "futures"
        symbols_list = futures_symbols
    
    # Build symbols dictionary based on asset type and symbols list
    symbols = {}
    
    if asset_type == "mixed":
        # Parse mixed symbols with format "SYMBOL:TYPE"
        for symbol_entry in symbols_list:
            if ':' in symbol_entry:
                symbol, symbol_type = symbol_entry.split(':', 1)
                symbols[symbol] = {
                    "asset_type": symbol_type.lower(),
                    "file_format": file_format
                }
            else:
                # Default to stocks if no type specified
                symbols[symbol_entry] = {
                    "asset_type": "stocks",
                    "file_format": file_format
                }
    else:
        # Single asset type
        for symbol in symbols_list:
            symbols[symbol] = {
                "asset_type": asset_type,
                "file_format": file_format
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
        slack_message = f"{status_emoji} **RCOV COMPUTATION COMPLETED** {status_emoji}\n\n"
        slack_message += f"**Status:** {status_text}\n"
        slack_message += f"**Pipeline:** {pipeline_name}\n"
        slack_message += f"**Asset Type:** {asset_type}\n"
        slack_message += f"**Duration:** {str(total_duration).split('.')[0]}\n"
        slack_message += f"**Symbols:** {len(symbols)}\n\n"
        
        # Add enabled asset types info
        enabled_types = []
        if stocks_enabled:
            enabled_types.append(f"stocks({len(stocks_symbols)})")
        if forex_enabled:
            enabled_types.append(f"forex({len(forex_symbols)})")
        if futures_enabled:
            enabled_types.append(f"futures({len(futures_symbols)})")
        if mixed_enabled:
            enabled_types.append(f"mixed({len(mixed_symbols)})")
        
        if enabled_types:
            slack_message += f"**Enabled Types:** {', '.join(enabled_types)}\n\n"
        
        slack_message += "**Processing Summary:**"
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
        sys.exit(1)


    