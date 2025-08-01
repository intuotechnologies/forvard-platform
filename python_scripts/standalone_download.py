"""
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics. 
Research code - Unauthorized distribution prohibited.

This script is a standalone downloader for Kibot data. It can be configured
to download data for various asset types and save it locally or to an
S3-compatible datalake.
"""

import os
import time
import queue
import logging
import threading
import concurrent.futures
import gc
import random
from datetime import datetime
from io import BytesIO, StringIO
from functools import wraps
from collections import OrderedDict, deque
import numpy as np
import pandas as pd
import requests
import psutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
import json
import sys
import argparse
import wmill
import s3fs


# --- Globals ---
print_lock = threading.Lock()
kibot_connection_semaphore = threading.BoundedSemaphore(4)
downloaded_symbols_registry = {}
downloaded_symbols_lock = threading.Lock()

# --- Logging and printing functions ---
def format_log_header(type, message):
    '''Format a header to make certain types of messages more visible'''
    if type == "PHASE":
        return f"\n{'='*30} {message} {'='*30}"
    elif type == "DOWNLOAD":
        return f"\n[DOWNLOAD] {'-'*10} {message} {'-'*10}"
    elif type == "PROCESS":
        return f"\n[PROCESS] {'-'*10} {message} {'-'*10}"
    elif type == "STEP":
        return f"\n  [STEP] {message}"
    elif type == "COMPLETE":
        return f"\n[COMPLETE] {'-'*10} {message} {'-'*10}"
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
        safe_print("Variabili d'ambiente Slack non trovate (SLACK_API_TOKEN, SLACK_CHANNEL_ID). Notifica saltata.", "WARNING")
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
        response.raise_for_status()
        
        response_json = response.json()
        if response_json.get("ok"):
            safe_print("Notifica di riepilogo inviata a Slack con successo.", "INFO")
        else:
            safe_print(f"Errore nell'invio a Slack: {response_json.get('error')}", "ERROR")

    except requests.exceptions.RequestException as e:
        safe_print(f"Errore di rete durante l'invio della notifica a Slack: {e}", "ERROR")
    except Exception as e:
        safe_print(f"Errore imprevisto durante l'invio a Slack: {e}", "ERROR")

# Logger configuration
logger = logging.getLogger("KibotDownloader")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class LRUCache:
    """Optimised LRU cache based on OrderedDict"""
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def __len__(self):
        with self.lock:
            return len(self.cache)

class PathManager:
    """Manages file paths in a centralised manner"""
    @staticmethod
    def ensure_directory_exists(path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod    
    def get_symbol_folder(base_dir, symbol, asset_type=None):
        if asset_type:
            return os.path.join(base_dir, asset_type, symbol)
        else:
            return os.path.join(base_dir, symbol)
    
    @staticmethod
    def ensure_symbol_folders(base_dir, symbols, asset_type=None):
        for symbol in symbols:
            os.makedirs(PathManager.get_symbol_folder(base_dir, symbol, asset_type), exist_ok=True)
    
    @staticmethod
    def get_date_filename(date):
        dt = datetime.strptime(date, "%m/%d/%Y")
        return dt.strftime("%Y_%m_%d")
    
    @staticmethod
    def get_data_filepath(base_dir, symbol, date, asset_type=None):
        date_filename = PathManager.get_date_filename(date)
        symbol_folder = PathManager.get_symbol_folder(base_dir, symbol, asset_type)    
        return os.path.join(symbol_folder, date_filename)
    
    @staticmethod  
    def get_adjustment_filepath(base_dir, symbol, asset_type=None):
        symbol_folder = PathManager.get_symbol_folder(base_dir, symbol, asset_type)
        return os.path.join(symbol_folder, "adjustment.txt")

class RateLimiter:
    """Advanced rate limiting management with automatic adjustment"""
    def __init__(self, max_per_second=10.0, min_rate=1.0, max_rate=50.0):
        self.current_rate = max_per_second
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.min_interval = 1.0 / self.current_rate
        self.last_called = 0
        self.lock = threading.RLock()
        self.consecutive_failures = 0
        self.consecutive_successes = 0
    
    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_called
            to_wait = max(0, self.min_interval - elapsed)
            if to_wait > 0:
                time.sleep(to_wait)
            self.last_called = time.time()
    
    def report_success(self):
        with self.lock:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            if self.consecutive_successes >= 5:
                self.current_rate = min(self.max_rate, self.current_rate * 1.1)
                self.min_interval = 1.0 / self.current_rate
                self.consecutive_successes = 0
    
    def report_failure(self, is_rate_limit=False):
        with self.lock:
            self.consecutive_successes = 0
            self.consecutive_failures += 1
            if is_rate_limit:
                self.current_rate = max(self.min_rate, self.current_rate * 0.5)
            else:
                self.current_rate = max(self.min_rate, self.current_rate * 0.8)
            self.min_interval = 1.0 / self.current_rate
            self.consecutive_failures = 0

class AdaptiveRateLimiter(RateLimiter):
    def __init__(self, max_per_second=10.0, min_rate=1.0, max_rate=20.0):
        super().__init__(max_per_second, min_rate, max_rate)
        self.recent_response_times = deque(maxlen=20)
        self.recent_success_flags = deque(maxlen=20)
    
    def report_success(self, response_time=None):
        with self.lock:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            if response_time:
                self.recent_response_times.append(response_time)
                self.recent_success_flags.append(True)
                if len(self.recent_response_times) >= 10:
                    avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times)
                    success_rate = sum(self.recent_success_flags) / len(self.recent_success_flags)
                    if success_rate > 0.95 and avg_response_time < 0.5:
                        self.current_rate = min(self.max_rate, self.current_rate * 1.2)
                    elif success_rate < 0.7 or avg_response_time > 2.0:
                        self.current_rate = max(self.min_rate, self.current_rate * 0.8)
                    self.min_interval = 1.0 / self.current_rate

class HTTPClient:
    """Unified HTTP client handling session, retry and rate limiting"""
    def __init__(self, max_connections=4, user_agent="KibotDownloader", 
                 connection_timeout=30, max_retries=3, 
                 rate_limiter=None):
        self.max_connections = max_connections
        self.user_agent = user_agent
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter(
            max_per_second=10.0, min_rate=2.0, max_rate=30.0
        )
        self.pool = []
        self.pool_lock = threading.RLock()
        self.connection_semaphore = threading.BoundedSemaphore(max_connections)
    
    def create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries, backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"], respect_retry_after_header=True,
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=2, pool_maxsize=4)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': self.user_agent, 'Accept-Encoding': 'gzip,deflate'
        })
        return session
    
    def get_session(self):
        with self.pool_lock:
            if self.pool:
                return self.pool.pop()
            return self.create_session()
    
    def release_session(self, session):
        with self.pool_lock:
            if len(self.pool) < self.max_connections:
                self.pool.append(session)
            else:
                session.close()
    
    def close_all(self):
        with self.pool_lock:
            for session in self.pool:
                try:
                    session.close()
                except:
                    pass
            self.pool = []
    
    def request(self, url, timeout=None, stream=False):
        with self.connection_semaphore:
            session = self.get_session()
            timeout = timeout or self.connection_timeout
            try:
                self.rate_limiter.wait()
                response = session.get(url, timeout=timeout, stream=stream)
                if response.status_code == 200:
                    self.rate_limiter.report_success()
                    return response, "success"
                elif response.status_code == 429:
                    self.rate_limiter.report_failure(is_rate_limit=True)
                    return None, f"rate_limit_{response.status_code}"
                else:
                    self.rate_limiter.report_failure()
                    return None, f"http_error_{response.status_code}"
            except requests.exceptions.ReadTimeout:
                self.rate_limiter.report_failure()
                return None, "timeout_error"
            except requests.exceptions.RequestException as e:
                self.rate_limiter.report_failure()
                return None, f"request_error: {str(e)}"
            except Exception as e:
                self.rate_limiter.report_failure()
                return None, f"error: {str(e)}"
            finally:
                self.release_session(session)

class KibotAPI:
    """Simplified and optimised Kibot API client"""
    BASE_URL = "http://api.kibot.com"
    
    def __init__(self, config, http_client=None):
        self.config = config
        self.http_client = http_client or HTTPClient(
            max_connections=config.max_concurrent_connections,
            connection_timeout=config.connection_timeout
        )
        self.response_cache = LRUCache(capacity=1000)
        self.login_timestamp = 0
        self.login_valid_duration = 15 * 60
        self.login_lock = threading.RLock()
        self.is_logged_in = False
    
    def ensure_login(self):
        return True

    def _login(self):
        return True

    def add_auth_to_url(self, url):
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}user={self.config.user}&password={self.config.pwd}"

    def download_data(self, symbol, date, max_attempts=3):
        if not self.ensure_login():
            return None, "login_failed"
        
        cache_key = f"{symbol}_{date}_{self.config.frequency}"
        cached_data = self.response_cache.get(cache_key)
        if cached_data is not None:
            return cached_data, "cached"
        
        base_url = (
            f"{self.BASE_URL}/?action=history&symbol={symbol}"
            f"&interval={self.config.frequency}&period=1&startdate={date}&enddate={date}"
            f"&regularsession=0&unadjusted=1&type={self.config.asset_type}"
        ) 
        url = self.add_auth_to_url(base_url)

        for attempt in range(max_attempts):
            response, status = self.http_client.request(url, timeout=self.config.data_timeout)
            
            if status != "success":
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Attempt {attempt+1}/{max_attempts} failed for {symbol} {date}. "
                            f"Waiting for {wait_time}s before the next attempt.")
                    time.sleep(wait_time)
                continue
            
            response_text = response.text
            
            if "405 Data Not Found" in response_text[:100]:
                return None, "no_data"
            
            if "401 Not Logged In" in response_text:
                logger.warning(f"Authentication problem for {symbol} {date}")
                self.is_logged_in = False
                if self._login():
                    if attempt < max_attempts - 1:
                        logger.info(f"Re-login successful, retry download for {symbol} {date}")
                        continue
                    else:
                        url = self.add_auth_to_url(base_url)
                        response, status = self.http_client.request(url, timeout=self.config.data_timeout)
                        if status == "success":
                            try:
                                df = pd.read_csv(BytesIO(response.content), header=None, dtype=self.config.datatype)
                                if df.empty: return None, "empty_data"
                                self.response_cache.put(cache_key, df)
                                return df, "success"
                            except Exception as parse_error:
                                logger.error(f"Error during data analysis for {symbol} {date}: {parse_error}")
                                return None, f"parse_error: {str(parse_error)}"
                return None, "auth_failed"
        
            try:
                df = pd.read_csv(BytesIO(response.content), header=None, dtype=self.config.datatype)
                if df.empty: return None, "empty_data"
                self.response_cache.put(cache_key, df)
                return df, "success"
            except Exception as parse_error:
                logger.error(f"Error during data analysis for {symbol} {date}: {parse_error}")
                return None, f"parse_error: {str(parse_error)}"
        
        return None, "max_retries_reached"
    
    def download_adjustment_data(self, symbol, max_attempts=3):
        if not self.ensure_login():
            return "login_failed"
        
        base_url = (f"{self.BASE_URL}?action=adjustments&symbol={symbol}"
                    f"&startdate={self.config.start_date}&enddate={self.config.end_date}")
        url = self.add_auth_to_url(base_url)
            
        for attempt in range(max_attempts):
            response, status = self.http_client.request(url)
            
            if status != "success":
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Tentativo {attempt+1}/{max_attempts} fallito per aggiustamento {symbol}. Attesa di {wait_time}s.")
                    time.sleep(wait_time)
                continue
            
            response_text = response.text
            if "405 Data Not Found" in response_text[:100]:
                return "no_data"
            
            if "401 Not Logged In" in response_text:
                self.is_logged_in = False
                if self._login():
                    return self.download_adjustment_data(symbol, max_attempts=1)
                return "session_expired"
            
            try:
                df = pd.read_csv(BytesIO(response.content), sep='\t')
                if df.empty: return "empty_data"
                path = PathManager.get_adjustment_filepath(self.config.base_dir, symbol, self.config.asset_type)
                PathManager.ensure_directory_exists(path)
                df.to_csv(path, sep='\t', index=False)
                return "success"
            except Exception as parse_error:
                logger.error(f"Error during analysis of adjustment data for {symbol}: {parse_error}")
                return f"parse_error: {str(parse_error)}"
        
        return "max_retries_reached"

class DataProcessor:
    """Processes downloaded data easily and efficiently"""
    COLUMN_CONFIGS = {
        'tickms': {'time': 1, 'price': 2, 'volume': 3},
        'tick': {'time': 1, 'price': 2, 'volume': 3},
        'tickbidaskms': {'time': 1, 'price': 2, 'bid': 3, 'ask': 4, 'size': 5},
        'tickbidask': {'time': 1, 'price': 2, 'bid': 3, 'ask': 4, 'size': 5}
    }
    
    @staticmethod
    def process_data(df, data_type, asset_type, chunk_size=None, retry_on_error=True, **kwargs):
        if df is None or df.empty:
            logger.warning("DataFrame empty or None, cannot process")
            return None
        try:
            if data_type not in DataProcessor.COLUMN_CONFIGS:
                logger.info(f"Data type ‘{data_type}’ not recognised, return base columns")
                return df.iloc[:, 1:]
            
            cols = DataProcessor.COLUMN_CONFIGS[data_type]
            time_col = cols['time']
            df['trades'] = df.groupby(time_col)[time_col].transform('count')
            df = df.sort_values(by=time_col)
            
            if asset_type and asset_type.lower() in ['forex', 'futures']:
                result = DataProcessor._process_forex_futures_data(df, cols)
            else:
                if 'volume' in cols:
                    result = DataProcessor._process_tick_data(df, cols)
                else:
                    result = DataProcessor._process_bidask_data(df, cols)
            return result
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}", exc_info=True)
            if retry_on_error:
                logger.info("Fallback attempt with simplified processing")
                try:
                    return DataProcessor._simple_process(df, data_type)
                except Exception as fallback_error:
                    logger.error(f"The fallback processing also failed: {str(fallback_error)}")
            return df.iloc[:, 1:]
    
    @staticmethod
    def _simple_process(df, data_type):
        cols = DataProcessor.COLUMN_CONFIGS.get(data_type, {})
        if not cols: return df.iloc[:, 1:]
        time_col = cols.get('time', 1)
        df = df.sort_values(by=time_col)
        return df
    
    @staticmethod
    def _process_tick_data(df, cols):
        time_col, price_col, volume_col = cols['time'], cols['price'], cols['volume']
        df['weight'] = df[volume_col] / df.groupby(time_col)[volume_col].transform('sum')
        df['weighted_price'] = df[price_col] * df['weight']
        agg_df = df.groupby(time_col).agg({'weighted_price': 'sum', volume_col: 'sum', 'trades': 'last'}).reset_index()
        agg_df.rename(columns={'weighted_price': price_col}, inplace=True)
        return agg_df
    
    @staticmethod
    def _process_bidask_data(df, cols):
        time_col, price_col, size_col, bid_col, ask_col = cols['time'], cols['price'], cols['size'], cols['bid'], cols['ask']
        df['weight'] = df[size_col] / df.groupby(time_col)[size_col].transform('sum')
        df['weighted_price'] = df[price_col] * df['weight']
        df['weighted_bid'] = df[bid_col] * df['weight']
        df['weighted_ask'] = df[ask_col] * df['weight']
        agg_columns = {'weighted_price': 'sum', 'weighted_bid': 'sum', 'weighted_ask': 'sum', size_col: 'sum', 'trades': 'last'}
        agg_df = df.groupby(time_col).agg(agg_columns).reset_index()
        agg_df.rename(columns={'weighted_price': price_col, 'weighted_bid': bid_col, 'weighted_ask': ask_col}, inplace=True)
        return agg_df
    
    @staticmethod
    def _process_forex_futures_data(df, cols):
        time_col, price_col, bid_col, ask_col, volume_col = cols['time'], cols['price'], cols['bid'], cols['ask'], cols['size']
        agg_columns = {price_col: 'median', bid_col: 'median', ask_col: 'median', volume_col: 'sum', 'trades': 'last'}
        agg_df = df.groupby(time_col).agg(agg_columns).reset_index()
        return agg_df

    @staticmethod
    def save_data(df, file_path, file_format='parquet', max_retries=3, retry_delay=2):
        """
        Save data to disk safely and efficiently with retry mechanism.
        This version saves *exclusively* to an S3-compatible datalake.
        """
        if df is None or df.empty:
            logger.warning(f"Attempt to save empty DataFrame for path {file_path}")
            return False

        # --- S3 Configuration fetched from Windmill/Env ---
        s3_endpoint_url = wmill.get_variable("u/niccolosalvini27/S3_ENDPOINT_URL")
        s3_access_key = wmill.get_variable("u/niccolosalvini27/S3_ACCESS_KEY")
        s3_secret_key = wmill.get_variable("u/niccolosalvini27/S3_SECRET_KEY")
        s3_bucket = wmill.get_variable("u/niccolosalvini27/S3_BUCKET")

        # Check for S3 configuration. If missing, fail explicitly.
        if not all([s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket]):
            error_msg = "S3 environment variables are not set. Cannot save to datalake. Please set S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, and S3_BUCKET."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not s3fs:
            error_msg = "The 's3fs' package is required to save to S3. Please install it."
            logger.error(error_msg)
            raise ImportError(error_msg)

        # Determine the extension and construct the full local-equivalent path
        extension = '.parquet' if file_format == 'parquet' else '.txt'
        final_path_local_equivalent = f"{file_path}{extension}"
        
        # Construct the S3 object key to mimic local directory structure
        base_dir = wmill.get_variable("u/niccolosalvini27/BASE_DIR").replace('\\', '/')
        # Ensure base_dir is not empty and is an absolute path for relpath
        if base_dir and os.path.isabs(final_path_local_equivalent):
             s3_object_key = os.path.relpath(final_path_local_equivalent, start=base_dir)
        else:
             s3_object_key = final_path_local_equivalent # Fallback for relative paths or missing base_dir
        
        # Replace backslashes with forward slashes for S3 compatibility, outside of the f-string.
        s3_object_key_safe = s3_object_key.replace('\\', '/')
        s3_full_path = f"{s3_bucket}/{s3_object_key_safe}"

        # Multiple rescue attempts
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt+1}/{max_retries} to save to S3 path: {s3_full_path}")
                
                # Initialize S3 filesystem object
                s3 = s3fs.S3FileSystem(
                    client_kwargs={'endpoint_url': s3_endpoint_url},
                    key=s3_access_key,
                    secret=s3_secret_key
                )

                # Save data with format-optimised parameters directly to S3
                if file_format == 'parquet':
                    with s3.open(s3_full_path, 'wb') as f:
                        df.to_parquet(f, compression='snappy', index=False)
                else: # txt/csv
                    csv_buffer = StringIO()
                    df.to_csv(
                        csv_buffer, 
                        float_format='%.15g',
                        date_format='%Y-%m-%d %H:%M:%S.%f',
                        index=False, 
                        header=False
                    )
                    with s3.open(s3_full_path, 'w') as f:
                        f.write(csv_buffer.getvalue())
                
                logger.info(f"Successfully saved to {s3_full_path}")
                return True
                
            except Exception as e:
                logger.error(f"S3 save attempt {attempt+1}/{max_retries} failed for {s3_full_path}: {str(e)}")
                
                # If not the last attempt, wait and try again
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting for {wait_time}s before the next S3 save attempt")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All S3 save attempts failed for {s3_full_path}")
        
        return False

    @staticmethod
    def verify_file(file_path, file_format='parquet'):
        extension = '.parquet' if file_format == 'parquet' else '.txt'
        full_path = f"{file_path}{extension}"
        if not os.path.exists(full_path): return False, f"File not found: {full_path}"
        if os.path.getsize(full_path) == 0: return False, f"Empty file: {full_path}"
        try:
            if file_format == 'parquet':
                metadata = pq.read_metadata(full_path)
                if metadata.num_rows == 0: return False, f"Readable but empty parquet file: {full_path}"
            else:
                df = pd.read_csv(full_path, nrows=1)
                if df is None or df.empty: return False, f"Readable but empty file: {full_path}"
            return True, f"Valid file: {full_path}"
        except Exception as e:
            return False, f"Error while reading file: {str(e)}"

class ProgressMonitor:
    def __init__(self, total_items, log_step=20):
        self.total_items = total_items
        self.completed_items = 0
        self.lock = threading.RLock()
        self.last_logged_percent = -1
        self.log_step = log_step
    
    def update(self):
        with self.lock:
            self.completed_items += 1
            percent = int((self.completed_items / self.total_items) * 100)
            if percent // self.log_step > self.last_logged_percent // self.log_step:
                logger.info(f"Progress: {percent}% completed ({self.completed_items}/{self.total_items})")
                self.last_logged_percent = percent
                return True
        return False

class KibotDownloader:
    """Optimised downloader for Kibot data"""
    def __init__(self, config):
        self.config = config
        self.http_client = HTTPClient(
            max_connections=min(config.max_concurrent_connections, 4),
            user_agent="KibotPythonClient", connection_timeout=config.connection_timeout
        )
        self.api = KibotAPI(config, self.http_client)
        self.existing_files = {}
        self.stats = defaultdict(int)
        self.lock = threading.RLock()
        self.processing_queue = queue.Queue(maxsize=100)
        self.retry_queue = queue.Queue()
        self.processing_done = threading.Event()
        self.processed_registry = set()
        self.pause_downloads = False
        self.processed_count = 0
        self.gc_frequency = 50
        self.item_status = defaultdict(dict)
        
    def check_memory_critically(self):
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            self.pause_downloads = True
            gc.collect(generation=2)
            self.api.response_cache.clear()
            for _ in range(5):
                gc.collect()
                if psutil.virtual_memory().percent < 85: break
                time.sleep(1)
            self.pause_downloads = False
        elif memory_percent > 80:
            gc.collect()

    def log_download_info(self):
        log_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = datetime.strptime(self.config.start_date, "%m/%d/%Y").strftime("%Y_%m_%d")
        end_date = datetime.strptime(self.config.end_date, "%m/%d/%Y").strftime("%Y_%m_%d")
        for symbol in self.config.symbols:
            path = os.path.join(
                PathManager.get_symbol_folder(self.config.base_dir, symbol, self.config.asset_type), 
                f"{symbol}_last_update.txt"
            )
            PathManager.ensure_directory_exists(path)
            with open(path, 'a') as f:
                f.write(f"{log_date}, {start_date}, {end_date}\n")
    
    def collect_existing_files(self):
        logger.info("Gathering information on existing files...")
        for symbol in self.config.symbols:
            folder_path = PathManager.get_symbol_folder(self.config.base_dir, symbol, self.config.asset_type)
            if not os.path.exists(folder_path):
                self.existing_files[symbol] = set()
                continue
            extension = '.parquet' if self.config.file_format == 'parquet' else '.txt'
            files = {file[:-len(extension)] for file in os.listdir(folder_path) if file.endswith(extension) and file not in ['adjustment.txt', f'{symbol}_last_update.txt']}
            self.existing_files[symbol] = files
            logger.debug(f"Found {len(files)} existing files for {symbol}")

    def is_file_existing(self, symbol, date):
        item_key = f"{symbol}_{date}"
        if item_key in self.processed_registry: return True
        date_filename = PathManager.get_date_filename(date)
        if symbol in self.existing_files and date_filename in self.existing_files[symbol]: return True
        
        # Check S3 if configured
        s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
        if s3_endpoint_url:
            # this check is expensive, so we rely on the local check if possible
            # or a separate manifest file in the future.
            # For now, we assume if local exists, s3 also exists if s3 is primary storage.
            pass

        file_path = PathManager.get_data_filepath(self.config.base_dir, symbol, date, self.config.asset_type)
        extension = '.parquet' if self.config.file_format == 'parquet' else '.txt'
        exists = os.path.exists(f"{file_path}{extension}")
        if exists:
            with self.lock:
                if symbol not in self.existing_files: self.existing_files[symbol] = set()
                self.existing_files[symbol].add(date_filename)
        return exists

    def generate_dates(self):
        start_dt = datetime.strptime(self.config.start_date, "%m/%d/%Y")
        end_dt = datetime.strptime(self.config.end_date, "%m/%d/%Y")
        exclude_we = self.config.asset_type not in ('forex', 'futures')
        if exclude_we:
            dates = pd.bdate_range(start_dt, end_dt)
        else:
            dates = pd.date_range(start_dt, end_dt)
        return [d.strftime("%m/%d/%Y") for d in dates] 
    
    def generate_work_items(self):
        dates = self.generate_dates()
        self.collect_existing_files()
        total_potential = len(self.config.symbols) * len(dates)
        logger.info(f"Analisi di {total_potential} potenziali file da scaricare...")
        work_items = []
        for symbol in self.config.symbols:
            missing_items = [(symbol, date) for date in dates if not self.is_file_existing(symbol, date)]
            work_items.extend(missing_items)
            if missing_items:
                logger.debug(f"Found {len(missing_items)} missing files for {symbol}")
        if len(work_items) > 100:
            random.shuffle(work_items)
        logger.info(f"Found {len(work_items)} missing files to download for {len({symbol for symbol, _ in work_items})} symbols")
        return work_items

    def _track_item_status(self, symbol, date, status):
        item_key = f"{symbol}_{date}"
        with self.lock:
            if status not in self.item_status[item_key]: self.item_status[item_key][status] = 0
            self.item_status[item_key][status] += 1
    
    def download_item(self, item):
        symbol, date = item
        item_key = f"{symbol}_{date}"
        if item_key in self.processed_registry: return "already_processed"
        if self.is_file_existing(symbol, date):
            with self.lock:
                self.stats["existing"] += 1
                self.stats["total"] += 1
            self._track_item_status(symbol, date, "existing")
            return "existing"
        
        df, status = self.api.download_data(symbol, date)
        self._track_item_status(symbol, date, status)

        if status == "auth_failed":
            logger.warning(f"Authentication failure for {symbol} {date}, immediate recovery attempt")
            self.api.is_logged_in = False
            if self.api._login():
                df, status = self.api.download_data(symbol, date)
                if status != "success":
                    self.retry_queue.put((symbol, date))
                    logger.error(f"Critical failure for {symbol} {date} after re-login. Added to retry queue.")
            else:
                logger.error(f"Unable to re-login for {symbol} {date}")
                self.retry_queue.put((symbol, date))
        
        self.progress_monitor.update()
        
        with self.lock:
            self.stats["total"] += 1
            if status == "success": self.stats["downloaded"] += 1
            elif status == "no_data": self.stats["no_data"] += 1
            elif status == "timeout_error":
                self.stats["timeout"] += 1
                self.stats["failed"] += 1
                self.stats["failed_downloads"] += 1
            elif not status.startswith("cached"):
                self.stats["failed"] += 1
                self.stats["failed_downloads"] += 1
        
        if status == "success" or status == "cached":
            try:
                self.processing_queue.put((symbol, date, df, 0), block=True, timeout=10)
                return "queued_for_processing"
            except queue.Full:
                logger.warning(f"Full processing queue. Immediate processing for {symbol} {date}")
                success = self._process_item(symbol, date, df)
                return "processed_immediately" if success else "processing_failed"
        return status

    def _process_item(self, symbol, date, df, attempt=0, max_attempts=3):
        item_key = f"{symbol}_{date}"
        if item_key in self.processed_registry: return True
        try:
            processed_df = (DataProcessor.process_data(df, self.config.frequency, self.config.asset_type)
                            if self.config.should_process_data else df.iloc[:, 1:])
            if processed_df is None or processed_df.empty:
                raise ValueError("DataFrame processed is empty")
            
            file_path = PathManager.get_data_filepath(self.config.base_dir, symbol, date, self.config.asset_type)
            success = DataProcessor.save_data(processed_df, file_path, self.config.file_format) 
            
            if success:
                self.processed_registry.add(item_key)
                with self.lock:
                    self.stats["processed"] += 1
                    date_filename = PathManager.get_date_filename(date)
                    self.existing_files.setdefault(symbol, set()).add(date_filename)
                return True
        except Exception as e:
            logger.error(f"Processing error {symbol} {date}: {e}")
        
        if attempt < max_attempts - 1:
            return self._process_item(symbol, date, df, attempt + 1, max_attempts)
        
        with self.lock:
            self.stats["failed"] += 1
            self.stats["failed_processing"] += 1
        return False

    def process_retry_queue(self):
        if self.retry_queue.empty():
            logger.info("No items in the retry queue")
            return
        
        retry_items = []
        while not self.retry_queue.empty():
            try:
                retry_items.append(self.retry_queue.get_nowait())
                self.retry_queue.task_done()
            except queue.Empty:
                break
        
        auth_failed_items = [item for item in retry_items if 'auth_failed' in self.item_status.get(f"{item[0]}_{item[1]}", {})]
        other_failed_items = [item for item in retry_items if item not in auth_failed_items]
        
        successful = 0
        for items, description in [(auth_failed_items, "authentication"), (other_failed_items, "other")]:
            if items:
                logger.info(f"Processing {len(items)} elements with {description} errors")
                for symbol, date in items:
                    try:
                        df, status = self.api.download_data(symbol, date)
                        self._track_item_status(symbol, date, f"retry_{status}")
                        if status == "success" or status == "cached":
                            if self._process_item(symbol, date, df):
                                successful += 1
                                with self.lock: self.stats["retry_success"] += 1
                    except Exception as e:
                        logger.error(f"Error during retry for {symbol} {date}: {e}")
        logger.info(f"Completed retry: {successful}/{len(retry_items)} processed successfully")

    def process_item_from_queue(self):
        while not self.processing_done.is_set():
            try:
                symbol, date, df, attempts = self.processing_queue.get(timeout=1)
                try:
                    max_attempts = 3
                    if attempts < max_attempts:
                        success = self._process_item(symbol, date, df, attempts, max_attempts)
                        if not success and attempts < max_attempts - 1:
                            self.processing_queue.put((symbol, date, df, attempts + 1))
                            with self.lock: self.stats["retried"] += 1
                    else:
                        logger.warning(f"Too many failed processing attempts for {symbol} {date}")
                        if self.config.enable_retry_queue:
                            self.retry_queue.put((symbol, date))
                            self._track_item_status(symbol, date, "retry_queued")
                except Exception as e:
                    logger.error(f"Error in worker during data processing for {symbol} {date}: {e}", exc_info=True)
                finally:
                    try: del df
                    except: pass
                    self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in the processing worker: {e}", exc_info=True)
                
    def run(self):
        start_time = time.time()
        try:
            os.makedirs(self.config.base_dir, exist_ok=True)
            PathManager.ensure_symbol_folders(self.config.base_dir, self.config.symbols, self.config.asset_type)
            
            today = datetime.now().strftime("%Y_%m_%d")
            log_dir = os.path.join(self.config.base_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"download_{today}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            
            logger.info(f"Start downloading for symbols={self.config.symbols}, " + 
                       f"start date={self.config.start_date}, end date={self.config.end_date}")
            
            self.log_download_info()
            dates = self.generate_dates()
            total_potential = len(self.config.symbols) * len(dates)
            work_items = self.generate_work_items()
            total_items = len(work_items)
            existing_files = total_potential - total_items
            with self.lock:
                self.stats["existing"] = existing_files
                self.stats["total"] += existing_files
            self.progress_monitor = ProgressMonitor(total_items, log_step=10)
    
            if total_items == 0:
                logger.info("No new data to download. All files are already existing.")
                if self.config.download_adjustments: self.download_adjustments()
                return True
            
            logger.info(f"Downloading {total_items} file for {len(self.config.symbols)} symbols...")
            
            self.processing_done.clear()
            processing_workers = [threading.Thread(target=self.process_item_from_queue, name=f"ProcessingWorker-{i}", daemon=True) 
                                  for i in range(self.config.max_processing_workers)]
            for worker in processing_workers: worker.start()
            
            chunk_size = min(500, total_items)
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.max_download_workers, 4)) as executor:
                for i in range(0, total_items, chunk_size):
                    while self.pause_downloads:
                        time.sleep(0.5)
                        self.check_memory_critically()
                    
                    chunk = work_items[i:i+chunk_size]
                    futures = [executor.submit(self.download_item, item) for item in chunk]
                    concurrent.futures.wait(futures)
                    self.check_memory_critically()
                    
                    while self.processing_queue.qsize() > 50:
                        logger.info(f"Processing queue full ({self.processing_queue.qsize()}), awaiting processing...")
                        time.sleep(2)
                        self.check_memory_critically()
            
            logger.info("Download complete. Waiting for processing to complete...")
            self.processing_queue.join()
            if self.config.enable_retry_queue and not self.retry_queue.empty():
                logger.info("Processing failed elements...")
                self.process_retry_queue()
            
            self.processing_done.set()
            for worker in processing_workers: worker.join(timeout=5)
            
            if self.config.download_adjustments:
                logger.info("Start downloading adjustment data...")
                self.download_adjustments()
                logger.info("Download adjustment data completed")
            
            total_time = time.time() - start_time
            total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
            
            # Replaced multiline f-string with individual log entries for better formatting
            logger.info("====== DOWNLOAD SUMMARY ======")
            logger.info(f"Completed in: {total_time_str}")
            logger.info(f"Total files analyzed: {self.stats['total']}")
            logger.info(f"  - Downloaded successfully: {self.stats['downloaded']}")
            logger.info(f"  - Already existed: {self.stats['existing']}")
            logger.info(f"  - No data on server: {self.stats['no_data']}")
            logger.info(f"  - Total failed: {self.stats['failed']}")
            logger.info(f"    - Download errors: {self.stats['failed_downloads']}")
            logger.info(f"    - Processing errors: {self.stats['failed_processing']}")
            logger.info(f"    - Connection timeouts: {self.stats['timeout']}")
            logger.info(f"Retried items: {self.stats['retried']}")
            logger.info(f"  - Successful recoveries: {self.stats['retry_success']}")
            logger.info(f"Files processed and saved: {self.stats['processed']}")
            logger.info(f"Average download speed: {self.stats['downloaded'] / max(1, total_time):.2f} files/s")
            logger.info(f"Final CPU usage: {psutil.cpu_percent()}%")
            logger.info(f"Final Memory usage: {psutil.virtual_memory().percent}%")
            logger.info("==============================")

            if self.config.generate_detailed_report: 
                self._generate_detailed_report()
                safe_print("Detailed report generated with failed dates for each tick", "INFO")
            return True
        except Exception as e:
            logger.error(f"Errore durante il download: {e}", exc_info=True)
            return False
        finally:
            self.processing_done.set()
            self.http_client.close_all()
    
    def _generate_detailed_report(self):
        """Generate a detailed report of the operation with symbol analysis and discrepancy detection"""
        try:
            log_dir = os.path.join(self.config.base_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            report_file = os.path.join(log_dir, f"report_{self.config.asset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            # Analizza i dati per simbolo
            symbol_stats = {}
            symbol_dates = {}
            
            for item_key, status_dict in self.item_status.items():
                # Estrae simbolo e data dall'item_key (formato: SYMBOL_MM/DD/YYYY)
                parts = item_key.split('_', 1)
                if len(parts) == 2:
                    symbol = parts[0]
                    date = parts[1]
                    
                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {'success': 0, 'no_data': 0, 'total': 0}
                        symbol_dates[symbol] = {'success': [], 'no_data': []}
                    
                    symbol_stats[symbol]['total'] += 1
                    
                    if 'success' in status_dict:
                        symbol_stats[symbol]['success'] += status_dict['success']
                        symbol_dates[symbol]['success'].append(date)
                    elif 'no_data' in status_dict:
                        symbol_stats[symbol]['no_data'] += status_dict['no_data']
                        symbol_dates[symbol]['no_data'].append(date)
            
            with open(report_file, 'w') as f:
                f.write("===== REPORT DETTAGLIATO =====\n\n")
                
                # Statistiche generali
                f.write("STATISTICHE GENERALI\n")
                for key, value in self.stats.items():
                    f.write(f"{key}: {value}\n")
                
                # Statistiche per simbolo
                f.write("\n===== STATISTICHE PER SIMBOLO =====\n")
                for symbol in sorted(symbol_stats.keys()):
                    stats = symbol_stats[symbol]
                    f.write(f"\n{symbol}:\n")
                    f.write(f"  success: {stats['success']}\n")
                    f.write(f"  no_data: {stats['no_data']}\n")
                    f.write(f"  total: {stats['total']}\n")
                
                # Controllo discrepanze tra simboli
                f.write("\n===== ANALISI DISCREPANZE =====\n")
                
                if len(symbol_stats) > 1:
                    symbols = list(symbol_stats.keys())
                    reference_symbol = symbols[0]
                    ref_success_dates = set(symbol_dates[reference_symbol]['success'])
                    ref_no_data_dates = set(symbol_dates[reference_symbol]['no_data'])
                    
                    discrepancies_found = False
                    
                    for symbol in symbols[1:]:
                        curr_success_dates = set(symbol_dates[symbol]['success'])
                        curr_no_data_dates = set(symbol_dates[symbol]['no_data'])
                        
                        # Trova differenze nei success
                        success_only_in_ref = ref_success_dates - curr_success_dates
                        success_only_in_curr = curr_success_dates - ref_success_dates
                        
                        # Trova differenze nei no_data
                        no_data_only_in_ref = ref_no_data_dates - curr_no_data_dates
                        no_data_only_in_curr = curr_no_data_dates - ref_no_data_dates
                        
                        if success_only_in_ref or success_only_in_curr or no_data_only_in_ref or no_data_only_in_curr:
                            discrepancies_found = True
                            f.write(f"\n WARNING: Discrepanze tra {reference_symbol} e {symbol}:\n")
                            
                            if success_only_in_ref:
                                f.write(f"  SUCCESS solo in {reference_symbol}: {sorted(success_only_in_ref)}\n")
                            if success_only_in_curr:
                                f.write(f"  SUCCESS solo in {symbol}: {sorted(success_only_in_curr)}\n")
                            if no_data_only_in_ref:
                                f.write(f"  NO_DATA solo in {reference_symbol}: {sorted(no_data_only_in_ref)}\n")
                            if no_data_only_in_curr:
                                f.write(f"  NO_DATA solo in {symbol}: {sorted(no_data_only_in_curr)}\n")
                    
                    if not discrepancies_found:
                        f.write(" Nessuna discrepanza trovata tra i simboli\n")
                else:
                    f.write("Un solo simbolo presente - controllo discrepanze non applicabile\n")
                
                # Dettaglio status per elemento (sezione originale)
                f.write("\n===== DETTAGLIO STATUS PER ELEMENTO =====\n")
                for item_key, status_dict in self.item_status.items():
                    f.write(f"{item_key}: {status_dict}\n")
            
            logger.info(f"Detailed report generated in {report_file}")
            
            # Log delle discrepanze anche nella console se trovate
            if len(symbol_stats) > 1:
                symbols = list(symbol_stats.keys())
                for i, symbol in enumerate(symbols):
                    stats = symbol_stats[symbol]
                    if i == 0:
                        continue
                    ref_stats = symbol_stats[symbols[0]]
                    if stats['success'] != ref_stats['success'] or stats['no_data'] != ref_stats['no_data']:
                        logger.warning(f"Discrepancy detected between {symbols[0]} and {symbol}")
                        
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def download_adjustments(self):
        logger.info(f"Download adjustment data for {len(self.config.symbols)} symbols")
        results = {}
        failed_symbols = []
        max_workers = min(self.config.max_download_workers, 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.api.download_adjustment_data, symbol): symbol for symbol in self.config.symbols}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    logger.info(f"Adjustment {symbol}: {result}")
                    self.check_memory_critically()
                except Exception as e:
                    logger.error(f"Error downloading adjustment for {symbol}: {e}")
                    results[symbol] = f"error: {str(e)}"
                    failed_symbols.append(symbol)
        
        if failed_symbols and self.config.retry_failed_adjustments:
            logger.info(f"Attempt to re-download fixes for {len(failed_symbols)} symbols failed")
            for symbol in failed_symbols:
                try:
                    time.sleep(1)
                    result = self.api.download_adjustment_data(symbol, max_attempts=2)
                    results[symbol] = result
                    logger.info(f"Retry adjustment {symbol}: {result}")
                except Exception as e:
                    logger.error(f"The retry also failed for {symbol}: {e}")
        
        success_count = sum(1 for result in results.values() if result == "success")
        logger.info(f"Download adjustments completed: {success_count}/{len(self.config.symbols)} riusciti")
        return results

class Config:
    """Kibot Downloader Setup with Advanced Parameters"""
    def __init__(self, user, pwd, base_dir, symbols, start_date, end_date, asset_type="stocks", frequency= None,
                 max_concurrent_connections=4, max_download_workers=4, max_processing_workers=4, connection_timeout=60,
                 data_timeout=120, download_adjustments=None, should_process_data=None, file_format="parquet",
                 chunk_size=50000, retry_count=4, verify_file_integrity=True, enable_retry_queue=True,
                 retry_failed_adjustments=True, generate_detailed_report=True):
        self.user, self.pwd = user, pwd
        self.base_dir, self.symbols, self.start_date, self.end_date = base_dir, symbols, start_date, end_date
        self.frequency, self.asset_type = frequency, asset_type

        if asset_type in ['stocks', 'ETFs']:
            self.frequency = frequency or "tickms"
            self.download_adjustments = download_adjustments if download_adjustments is not None else True
            self.should_process_data = should_process_data if should_process_data is not None else True
        elif asset_type in ['forex', 'futures']:
            self.frequency = frequency or "tickbidaskms"
            self.download_adjustments = download_adjustments if download_adjustments is not None else False
            self.should_process_data = should_process_data if should_process_data is not None else True
        else:
            self.frequency = frequency or "tickms"
            self.download_adjustments = download_adjustments if download_adjustments is not None else True
            self.should_process_data = should_process_data if should_process_data is not None else True
            
        self.max_concurrent_connections = min(max_concurrent_connections, 4)
        self.max_download_workers = min(max_download_workers, self.max_concurrent_connections)
        self.max_processing_workers = max_processing_workers
        self.connection_timeout, self.data_timeout = connection_timeout, data_timeout
        self.file_format = file_format.lower()
        if self.file_format not in ['txt', 'parquet']:
            logger.warning(f"File format '{file_format}' not supported. Using 'txt' as default.")
            self.file_format = 'txt'
        self.chunk_size, self.retry_count = chunk_size, retry_count
        self.verify_file_integrity, self.enable_retry_queue = verify_file_integrity, enable_retry_queue
        self.retry_failed_adjustments, self.generate_detailed_report = retry_failed_adjustments, generate_detailed_report
        
        self.datatype = {0: np.str_, 1: np.str_, 2: np.float64}
        if self.frequency.startswith('tick'):
            if 'bidask' in self.frequency:
                self.datatype.update({3: np.float64, 4: np.float64, 5: np.int64})
            else:
                self.datatype.update({3: np.int64})
    
    def __str__(self):
        return (f"Config(user='{self.user}', base_dir='{self.base_dir}', symbols=[{len(self.symbols)} simboli], "
                f"periodo={self.start_date}-{self.end_date}, frequency='{self.frequency}', asset_type='{self.asset_type}', "
                f"max_connections={self.max_concurrent_connections}, file_format={self.file_format})")

# --- Functions from main_pipeline.py ---

def load_config():
    """
    Loads configuration directly from a hardcoded Python dictionary.
    Environment variables for credentials and base directory are now fetched
    using the global get_variable helper function.
    """
    # load_dotenv() # No longer needed
    
    # Resolve environment variables
    kibot_user = wmill.get_variable("u/niccolosalvini27/KIBOT_USER")
    kibot_password = wmill.get_variable("u/niccolosalvini27/KIBOT_PASSWORD")
    base_dir = wmill.get_variable("u/niccolosalvini27/BASE_DIR").replace('\\', '/') # Standard env var

    # The entire configuration is now a Python dictionary
    config = {
    "credentials": {
        "kibot_user": kibot_user,
        "kibot_password": kibot_password
    },
    "general": {
        "base_dir": f"{base_dir}/data",
        "file_format": "parquet",
        "execution_mode": "async",
        "system_cores_reserved": 1,
        "download_threads_reserved": 4,
        "processing_threads_reserved": 4,
        "outliers_threads_max": 8,
        "rv_threads_max": 8
    },
    "pipelines": {
        "stocks_batch1": {
            "enabled": False,
            "asset_type": "stocks",
            "symbols": [
                "MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY",
                "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM",
                "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA",
                "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC",
                "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST",
                "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM",
                "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK",
                "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO",
                "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC"
            ],
            "date_range": {
                "start_date": "01/01/2015",
                "end_date": "07/17/2025"
            },
            "steps": ["download", "outliers", "realized_variance"]
        },
        "ETFs": {
            "enabled": True,
            "asset_type": "ETFs",
            "symbols": [
                "AGG", "BND", "GLD", "SLV", "SUSA", "EFIV", "ESGV", "ESGU", "AFTY", "MCHI", "EWH", "EEM",
                "IEUR", "VGK", "FLCH", "EWJ", "NKY", "EWZ", "EWC", "EWU", "EWI", "EWP", "ACWI", "IOO", "GWL",
                "VEU", "IJH", "MDY", "IVOO", "IYT", "XTN", "XLI", "XLU", "VPU", "SPSM", "IJR", "VIOO", "QQQ",
                "ICLN", "ARKK", "SPLG", "SPY", "VOO", "IYY", "VTI", "DIA"
            ],
            "date_range": {
                "start_date": "01/01/2015",
                "end_date": "07/17/2025"
            },
            "steps": ["download", "outliers", "realized_variance"]
        },
        "forex": {
            "enabled": True,
            "asset_type": "forex",
            "symbols": [
                "EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "JPYUSD", "CHFUSD",
                "SGDUSD", "HKDUSD", "KRWUSD", "INRUSD", "RUBUSD", "BRLUSD"
            ],
            "date_range": {
                "start_date": "01/01/2015",
                "end_date": "07/17/2025"
            },
            "steps": ["download", "realized_variance"]
        },
        "futures": {
            "enabled": True,
            "asset_type": "futures",
            "symbols": [
                "ES", "CL", "GC", "NG", "NQ", "TY", "FV", "EU", "SI", "C", "W", "VX"
            ],
            "date_range": {
                "start_date": "01/01/2015",
                "end_date": "07/17/2025"
            },
            "steps": ["download", "realized_variance"]
        }
    }
    }

            
    return config

def clean_kibot_logger():
    kibot_logger = logging.getLogger("KibotDownloader")
    if not kibot_logger.handlers: return
    while len(kibot_logger.handlers) > 1:
        handler = kibot_logger.handlers[-1]
        kibot_logger.removeHandler(handler)
        try: handler.close()
        except: pass
    kibot_logger.propagate = False

def is_symbol_already_downloaded(symbol, date, asset_type):
    with downloaded_symbols_lock:
        return f"{symbol}_{date}_{asset_type}" in downloaded_symbols_registry

def mark_symbol_as_downloaded(symbol, date, asset_type, status=True):
    with downloaded_symbols_lock:
        downloaded_symbols_registry[f"{symbol}_{date}_{asset_type}"] = status

def download_worker(config_data, pipeline_name, symbols_batch, result_queue):
    pipeline_config = config_data['pipelines'][pipeline_name]
    asset_type = pipeline_config['asset_type']
    date_range = pipeline_config['date_range']
    
    # Get credentials and base_dir from environment
    kibot_user = wmill.get_variable("u/niccolosalvini27/KIBOT_USER")
    kibot_password = wmill.get_variable("u/niccolosalvini27/KIBOT_PASSWORD")
    base_dir = wmill.get_variable("u/niccolosalvini27/BASE_DIR").replace('\\', '/')
    
    symbols_to_download = [
        s for s in symbols_batch 
        if not is_symbol_already_downloaded(s, date_range['start_date'], asset_type) and
           not is_symbol_already_downloaded(s, date_range['end_date'], asset_type)
    ]

    for s in symbols_to_download:
        mark_symbol_as_downloaded(s, date_range['start_date'], asset_type)
        mark_symbol_as_downloaded(s, date_range['end_date'], asset_type)

    if not symbols_to_download:
        safe_print(f"No new symbols to download for {pipeline_name}, all already processed", "INFO")
        result_queue.put({'pipeline': pipeline_name, 'symbols': symbols_batch, 'status': True, 'asset_type': asset_type})
        return True
    
    safe_print(f"Batch download of {len(symbols_to_download)}/{len(symbols_batch)} symbols for {pipeline_name}", "DOWNLOAD")
    
    with kibot_connection_semaphore:
        try:
            clean_kibot_logger()
            download_config = Config(
                user=kibot_user,
                pwd=kibot_password,
                base_dir=f"{base_dir}/data",
                symbols=symbols_to_download,
                start_date=date_range['start_date'],
                end_date=date_range['end_date'],
                asset_type=asset_type,
                file_format=config_data['general']['file_format']
            )
            downloader = KibotDownloader(download_config)
            success = downloader.run()
            
            # Return detailed stats
            stats_snapshot = downloader.stats.copy()
            stats_snapshot['pipeline'] = pipeline_name
            stats_snapshot['symbols_in_batch'] = len(symbols_batch)
            stats_snapshot['symbols_processed'] = len(symbols_to_download)
            
            result_queue.put({'pipeline': pipeline_name, 'symbols': symbols_batch, 'status': success, 'asset_type': asset_type, 'stats': stats_snapshot})
            safe_print(f"Download completed batch of {len(symbols_to_download)} symbols for {pipeline_name}", "COMPLETE")
            return success
        except Exception as e:
            safe_print(f"Batch download error of {pipeline_name}: {e}", "ERROR")
            for s in symbols_to_download:
                mark_symbol_as_downloaded(s, date_range['start_date'], asset_type, False)
                mark_symbol_as_downloaded(s, date_range['end_date'], asset_type, False)
            return False

# --- Main execution block ---
def main(
    pipelines=None,
    file_format="parquet",
    download_threads_max=4,
    processing_threads_max=4,
    stocks_enabled=True,
    stocks_symbols=["MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY",
                "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM",
                "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA",
                "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC",
                "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST",
                "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM",
                "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK",
                "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO",
                "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC"],
    etfs_enabled=True,
    etfs_symbols=["AGG", "BND", "GLD", "SLV", "SUSA", "EFIV", "ESGV", "ESGU", "AFTY", "MCHI", "EWH", "EEM",
                "IEUR", "VGK", "FLCH", "EWJ", "NKY", "EWZ", "EWC", "EWU", "EWI", "EWP", "ACWI", "IOO", "GWL",
                "VEU", "IJH", "MDY", "IVOO", "IYT", "XTN", "XLI", "XLU", "VPU", "SPSM", "IJR", "VIOO", "QQQ",
                "ICLN", "ARKK", "SPLG", "SPY", "VOO", "IYY", "VTI", "DIA"],
    forex_enabled=True,
    forex_symbols=["EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "JPYUSD", "CHFUSD",
                "SGDUSD", "HKDUSD", "KRWUSD", "INRUSD", "RUBUSD", "BRLUSD"],
    futures_enabled=True,
    futures_symbols=["ES", "CL", "GC", "NG", "NQ", "TY", "FV", "EU", "SI", "C", "W", "VX"],
    start_date="01/01/2015",
    end_date="07/17/2025",
    verbose=False
):
    """
    Standalone Data Downloader for ForVARD Project
    
    Args:
        pipelines: List of pipelines to run (e.g., ['stocks', 'ETFs'])
        file_format: File format to process ('parquet' or 'txt')
        download_threads_max: Maximum number of threads for download processing
        processing_threads_max: Maximum number of threads for data processing
        stocks_enabled: Enable stocks pipeline
        stocks_symbols: List of symbols for stocks
        etfs_enabled: Enable ETFs pipeline
        etfs_symbols: List of symbols for ETFs
        forex_enabled: Enable forex pipeline
        forex_symbols: List of symbols for forex
        futures_enabled: Enable futures pipeline
        futures_symbols: List of symbols for futures
        start_date: Start date for data download (MM/DD/YYYY format)
        end_date: End date for data download (MM/DD/YYYY format)
        verbose: Enable verbose logging
    """
    start_time = datetime.now()
    safe_print(f"PIPELINE START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", "PHASE")
    
    try:
        # Build configuration from arguments
        config = {
            "general": {
                "file_format": file_format,
                "download_threads_max": download_threads_max,
                "processing_threads_max": processing_threads_max
            },
            "pipelines": {
                "stocks": {
                    "enabled": stocks_enabled,
                    "asset_type": "stocks",
                    "symbols": stocks_symbols,
                    "date_range": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "steps": ["download"]
                },
                "ETFs": {
                    "enabled": etfs_enabled,
                    "asset_type": "ETFs",
                    "symbols": etfs_symbols,
                    "date_range": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "steps": ["download"]
                },
                "forex": {
                    "enabled": forex_enabled,
                    "asset_type": "forex",
                    "symbols": forex_symbols,
                    "date_range": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "steps": ["download"]
                },
                "futures": {
                    "enabled": futures_enabled,
                    "asset_type": "futures",
                    "symbols": futures_symbols,
                    "date_range": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "steps": ["download"]
                }
            }
        }
        
        # Determine which pipelines to run
        if pipelines:
            pipelines_to_run = pipelines
        else:
            pipelines_to_run = [name for name, conf in config['pipelines'].items() if conf.get('enabled', True)]
        
        if not pipelines_to_run:
            safe_print("No pipeline to run. Check configuration.", "ERROR")
            return
        
        safe_print(f"Pipelines to execute: {pipelines_to_run}", "INFO")
        
        results = {}
        for pipeline_name in pipelines_to_run:
            if pipeline_name not in config['pipelines'] or not config['pipelines'][pipeline_name].get('enabled', True):
                safe_print(f"Pipeline '{pipeline_name}' disabled or not found. Skipping.", "INFO")
                continue
            
            pipeline_config = config['pipelines'][pipeline_name]
            safe_print(f"Starting pipeline {pipeline_name.upper()}", "PHASE")
            safe_print(f"Symbols: {len(pipeline_config['symbols'])}, Type: {pipeline_config['asset_type']}", "INFO")
            
            clean_kibot_logger()
            
            all_symbols = pipeline_config['symbols']
            batch_size = max(15, len(all_symbols) // (os.cpu_count() or 4)) if len(all_symbols) >= 50 else \
                         max(10, len(all_symbols) // 3) if len(all_symbols) >= 20 else \
                         max(5, len(all_symbols) // 2) if len(all_symbols) > 10 else len(all_symbols)
            
            safe_print(f"Batch size: {batch_size} symbols", "INFO")
            symbol_batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
            
            result_queue = queue.Queue()
            all_stats = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=download_threads_max) as executor:
                futures = [executor.submit(download_worker, config, pipeline_name, batch, result_queue) for batch in symbol_batches]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    # This part ensures we wait for all workers to finish
                    # but the main logic for collecting stats is from the queue
                    pass

                pipeline_success = all(f.result() for f in futures)
                results[pipeline_name] = pipeline_success

            # Drain the queue to get all stats
            while not result_queue.empty():
                try:
                    all_stats.append(result_queue.get_nowait())
                except queue.Empty:
                    break
        
        end_time = datetime.now()
        duration = end_time - start_time

        safe_print("FINAL SUMMARY", "PHASE")
        safe_print(f"Total time: {duration}")
        
        for pipeline, success in results.items():
            safe_print(f"{pipeline:>10}: {'SUCCESS' if success else 'FAILED'}")
        
        # --- SLACK NOTIFICATION ---
        successful_pipelines = sum(1 for success in results.values() if success)
        failed_pipelines = len(results) - successful_pipelines
        
        if all(results.values()):
            safe_print("\nALL PIPELINES SUCCESSFULLY COMPLETED!", "PHASE")
            status_emoji = "✅"
            status_text = "SUCCESS"
        else:
            safe_print("\nSOME PIPELINES FAILED!", "PHASE")
            status_emoji = "⚠️" if failed_pipelines < successful_pipelines else "❌"
            status_text = "PARTIAL FAILURE" if successful_pipelines > 0 else "FAILURE"

        # Aggregate stats for the message
        total_downloaded = sum(s['stats'].get('downloaded', 0) for s in all_stats)
        total_existing = sum(s['stats'].get('existing', 0) for s in all_stats)
        total_no_data = sum(s['stats'].get('no_data', 0) for s in all_stats)
        total_failed = sum(s['stats'].get('failed', 0) for s in all_stats)
        total_processed = sum(s['stats'].get('processed', 0) for s in all_stats)

        message_parts = [
            f"{status_emoji} *KIBOT DOWNLOAD COMPLETED* {status_emoji}",
            "",
            f"*Status:* {status_text}",
            f"*Duration:* {str(duration).split('.')[0]}",
            f"*Pipelines:* {successful_pipelines}/{len(results)} successful",
            "",
            "*Pipeline Results:*"
        ]

        for pipeline, success in results.items():
            result_emoji = "✅" if success else "❌"
            message_parts.append(f"{result_emoji} {pipeline}: {'SUCCESS' if success else 'FAILED'}")

        message_parts.extend([
            "",
            "*Download Summary:*",
            f"✅ Downloaded successfully: {total_downloaded}",
            f"⏭️ Already existed: {total_existing}",
            f"❓ No data on server: {total_no_data}",
            f"❌ Total failed: {total_failed}",
            f"💾 Files processed and saved: {total_processed}"
        ])

        message_parts.extend([
            "",
            f"*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ])

        slack_message = "\n".join(message_parts)
        
        try:
            send_slack_notification(slack_message)
        except Exception as e:
            safe_print(f"Error sending Slack notification: {e}", "WARNING")

    except Exception as e:
        safe_print(f"CRITICAL ERROR: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 