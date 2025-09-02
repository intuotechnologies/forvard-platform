"""
ForVARD Project - Windmill-Compatible Kibot Data Download
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics. 
Research code - Unauthorized distribution prohibited.

Windmill-compatible version of the original download.py script.
Preserves IDENTICAL logic and logging but adapts for Windmill execution.
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
from io import BytesIO
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

# Windmill imports
import wmill

# S3/MinIO imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from io import StringIO

# Logger configuration - IDENTICAL to original
logger = logging.getLogger("KibotDownloader")
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
    """Handles writing downloaded data to S3-compatible storage (MinIO)"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("KibotDownloader")
        
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
    
    def save_data_to_s3(self, df, symbol, date, asset_type, file_format='parquet'):
        """Save processed data to S3 - IDENTICAL structure to local save"""
        if df is None or df.empty:
            self.logger.warning(f"Attempt to save empty DataFrame for {symbol} {date}")
            return False
        
        try:
            # Create S3 key following the same structure as local paths
            date_filename = PathManager.get_date_filename(date)
            
            if file_format == 'parquet':
                extension = '.parquet'
            else:
                extension = '.txt'
            
            # S3 key: data/{asset_type}/{symbol}/{date_filename}.{extension}
            s3_key = f"data/{asset_type}/{symbol}/{date_filename}{extension}"
            
            self.logger.debug(f"Saving to S3: {s3_key}")
            
            # Prepare data for upload
            if file_format == 'parquet':
                # Save as parquet
                import io
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
            else:
                # Save as CSV/TXT
                csv_buffer = StringIO()
                df.to_csv(
                    csv_buffer,
                    float_format='%.15g',  # 15 significant digits (max for float64)
                    date_format='%Y-%m-%d %H:%M:%S.%f',  # Microsecond precision
                    index=False,  # Include index
                    header=False,  # Include headers 
                )
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=csv_buffer.getvalue(),
                    ContentType='text/csv'
                )
            
            self.logger.debug(f"Successfully saved {symbol} {date} to S3: {s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {symbol} {date} to S3: {e}")
            return False
    
    def save_adjustment_data_to_s3(self, df, symbol, asset_type):
        """Save adjustment data to S3"""
        if df is None or df.empty:
            self.logger.warning(f"Attempt to save empty adjustment DataFrame for {symbol}")
            return False
        
        try:
            # S3 key for adjustment data
            s3_key = f"data/{asset_type}/{symbol}/adjustment.txt"
            
            # Save as TSV (tab-separated)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, sep='\t', index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/tab-separated-values'
            )
            
            self.logger.debug(f"Successfully saved adjustment data for {symbol} to S3: {s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving adjustment data for {symbol} to S3: {e}")
            return False
    
    def file_exists_in_s3(self, symbol, date, asset_type, file_format='parquet'):
        """Check if file already exists in S3"""
        try:
            date_filename = PathManager.get_date_filename(date)
            extension = '.parquet' if file_format == 'parquet' else '.txt'
            s3_key = f"data/{asset_type}/{symbol}/{date_filename}{extension}"
            
            # Check if object exists
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                self.logger.error(f"Error checking S3 file existence for {symbol} {date}: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Error checking S3 file existence for {symbol} {date}: {e}")
            return False
    
    def save_last_update_info_to_s3(self, symbol, asset_type, log_info):
        """Save last update info to S3"""
        try:
            s3_key = f"data/{asset_type}/{symbol}/{symbol}_last_update.txt"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=log_info,
                ContentType='text/plain'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving last update info for {symbol} to S3: {e}")
            return False

# ============================================================
# EXACT COPY OF ORIGINAL DOWNLOAD.PY CLASSES
# ============================================================

class LRUCache:
    """Optimised LRU cache based on OrderedDict"""
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move item to top (newest)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key, value):
        with self.lock:
            # If the key is already present, remove it to update it
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Remove the oldest element if necessary
                self.cache.popitem(last=False)
            
            # Add the new item (it will be the most recent)
            self.cache[key] = value
    
    def clear(self):
        """Empty the cache completely"""
        with self.lock:
            self.cache.clear()
    
    def __len__(self):
        """Returns the number of elements in the cache"""
        with self.lock:
            return len(self.cache)

class PathManager:
    """Manages file paths in a centralised manner"""
    @staticmethod
    def ensure_directory_exists(path):
        """Ensures that the directory for a file path exists"""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod    
    def get_symbol_folder(base_dir, symbol, asset_type=None):
        """Returns the folder path for a symbol organised by asset type"""
        if asset_type:
            return os.path.join(base_dir, asset_type, symbol)
        else:
            return os.path.join(base_dir, symbol)
    
    @staticmethod
    def ensure_symbol_folders(base_dir, symbols, asset_type=None):
        """Create folders for all symbols organised by asset type"""
        for symbol in symbols:
            os.makedirs(PathManager.get_symbol_folder(base_dir, symbol, asset_type), exist_ok=True)

    @staticmethod
    def get_date_filename(date):
        """Converts a date to file name format"""
        dt = datetime.strptime(date, "%m/%d/%Y")
        return dt.strftime("%Y_%m_%d")
    
    @staticmethod
    def get_data_filepath(base_dir, symbol, date, file_format=None, asset_type=None):
        """Returns the basic file path for data without extension"""
        date_filename = PathManager.get_date_filename(date)
        symbol_folder = PathManager.get_symbol_folder(base_dir, symbol, asset_type)    
        return os.path.join(symbol_folder, date_filename)
    
    @staticmethod  
    def get_adjustment_filepath(base_dir, symbol, asset_type=None):
        """Returns the file path for the adjustment data"""
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
        """It waits as long as it takes to meet the rate limit"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_called
            to_wait = max(0, self.min_interval - elapsed)
            
            if to_wait > 0:
                time.sleep(to_wait)
            
            self.last_called = time.time()
    
    def report_success(self):
        """Report a successful request to adjust the rate"""
        with self.lock:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            
            # Gradually increases speed after 5 consecutive successes
            if self.consecutive_successes >= 5:
                self.current_rate = min(self.max_rate, self.current_rate * 1.1)
                self.min_interval = 1.0 / self.current_rate
                self.consecutive_successes = 0
    
    def report_failure(self, is_rate_limit=False):
        """Report a failure to reduce the rate"""
        with self.lock:
            self.consecutive_successes = 0
            self.consecutive_failures += 1
            
            # Rate reduction strategy based on error type
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
        """Report a successful request with response time"""
        with self.lock:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            
            if response_time:
                self.recent_response_times.append(response_time)
                self.recent_success_flags.append(True)
                
                # Adjusts the rate according to recent performance
                if len(self.recent_response_times) >= 10:
                    avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times)
                    success_rate = sum(self.recent_success_flags) / len(self.recent_success_flags)
                    
                    if success_rate > 0.95 and avg_response_time < 0.5:
                        # Excellent performance, rate increases
                        self.current_rate = min(self.max_rate, self.current_rate * 1.2)
                    elif success_rate < 0.7 or avg_response_time > 2.0:
                        # Poor performance, lower the rate
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
        
        # Integrated rate limiter
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter(
            max_per_second=10.0,  # More aggressive initial value
            min_rate=2.0,        # Minimum
            max_rate=30.0        # Maximum
        )
        
        # Pool of sessions
        self.pool = []
        self.pool_lock = threading.RLock()
        
        # Semaphore to limit competing connections
        self.connection_semaphore = threading.BoundedSemaphore(max_connections)
    
    def create_session(self):
        """Create a new optimised HTTP session"""
        session = requests.Session()
        
        # Configura retry policy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=2,
            pool_maxsize=4
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip,deflate'
        })
        
        return session
    
    def get_session(self):
        """Obtains a session from the pool or creates a new one"""
        with self.pool_lock:
            if self.pool:
                return self.pool.pop()
            return self.create_session()
    
    def release_session(self, session):
        """Returns a session to the pool"""
        with self.pool_lock:
            if len(self.pool) < self.max_connections:
                self.pool.append(session)
            else:
                session.close()
    
    def close_all(self):
        """Closing all sessions in the pool"""
        with self.pool_lock:
            for session in self.pool:
                try:
                    session.close()
                except:
                    pass
            self.pool = []
    
    def request(self, url, timeout=None, stream=False):
        """Executes an HTTP request with automatic session handling, rate limiting and retry"""
        with self.connection_semaphore:
            session = self.get_session()
            timeout = timeout or self.connection_timeout
            
            try:
                # Apply rate limiting
                self.rate_limiter.wait()
                
                # Execute request
                response = session.get(url, timeout=timeout, stream=stream)
                
                # Manages replies according to status code
                if response.status_code == 200:
                    self.rate_limiter.report_success()
                    return response, "success"
                elif response.status_code == 429:  # Rate limit
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
    
    def __init__(self, config, http_client=None, s3_processor=None):
        self.config = config
        self.s3_processor = s3_processor
        self.http_client = http_client or HTTPClient(
            max_connections=config.max_concurrent_connections,
            connection_timeout=config.connection_timeout
        )
        self.response_cache = LRUCache(capacity=1000)
        self.login_timestamp = 0
        self.login_valid_duration = 15 * 60  # 15 minuti
        self.login_lock = threading.RLock()
        self.is_logged_in = False
    
    def ensure_login(self):
        """Fallback per situazioni in cui l'autenticazione nell'URL non funziona"""
        # With authentication in the URL, we should always be 'logged in'.
        return True
    
    def _login(self):
        """Simplified login method for rare error cases"""
        return True
    
    def add_auth_to_url(self, url):
        """Adds authentication parameters to the URL"""
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}user={self.config.user}&password={self.config.pwd}"

    def download_data(self, symbol, date, max_attempts=3):
        """Download data for a symbol and a date with authentication in the URL"""
        # Checking Login Status
        if not self.ensure_login():
            return None, "login_failed"
        
        # Checking the Cache
        cache_key = f"{symbol}_{date}_{self.config.frequency}"
        cached_data = self.response_cache.get(cache_key)
        if cached_data is not None:
            return cached_data, "cached"
        
        # Construct the base URL (combines usr and pwd in the call)
        base_url = (
            f"{self.BASE_URL}/?action=history&symbol={symbol}"
            f"&interval={self.config.frequency}&period=1&startdate={date}&enddate={date}"
            f"&regularsession=0&unadjusted=1&type={self.config.asset_type}"
        ) 
        # Add credentials to URL
        url = self.add_auth_to_url(base_url)
        
        for attempt in range(max_attempts):
            response, status = self.http_client.request(url, timeout=self.config.data_timeout)
            
            # Response management as before
            if status != "success":
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Attempt {attempt+1}/{max_attempts} failed for {symbol} {date}. "
                            f"Waiting for {wait_time}s before the next attempt.")
                    time.sleep(wait_time)
                continue
            
            # Analyses the answer
            response_text = response.text
            
            # Check if we have no data
            if "405 Data Not Found" in response_text[:100]:
                return None, "no_data"
            
            # There is no longer a need to check '401 Not Logged In' because the authentication is in the URL
            # but we keep it for robustness
            if "401 Not Logged In" in response_text:
                logger.warning(f"Authentication problem for {symbol} {date}")
                # Resetting the login to force a new login
                self.is_logged_in = False
                
                # Login again
                if self._login():
                    # Try the download again after re-login
                    if attempt < max_attempts - 1:
                        logger.info(f"Re-login successful, retry download for {symbol} {date}")
                        continue
                    else:
                        # Last attempt with a new URL after re-login
                        url = self.add_auth_to_url(base_url)
                        response, status = self.http_client.request(url, timeout=self.config.data_timeout)
                        if status == "success":
                            # Proceed with the processing of the answer as before
                            try:
                                df = pd.read_csv(
                                    BytesIO(response.content),
                                    header=None,
                                    dtype=self.config.datatype
                                )
                                
                                if df.empty:
                                    return None, "empty_data"
                                
                                self.response_cache.put(cache_key, df)
                                return df, "success"
                            except Exception as parse_error:
                                logger.error(f"Error during data analysis for {symbol} {date}: {parse_error}")
                                return None, f"parse_error: {str(parse_error)}"
                return None, "auth_failed"
        
            # Success - process data
            try:
                df = pd.read_csv(
                    BytesIO(response.content),
                    header=None,
                    dtype=self.config.datatype
                )
                
                if df.empty:
                    return None, "empty_data"
                
                # Cache
                self.response_cache.put(cache_key, df)
                return df, "success"
                
            except Exception as parse_error:
                logger.error(f"Error during data analysis for {symbol} {date}: {parse_error}")
                return None, f"parse_error: {str(parse_error)}"
        
        # If we get here, all attempts have failed
        return None, "max_retries_reached"
    
    def download_adjustment_data(self, symbol, max_attempts=3):
        """Download adjustment data for a symbol"""
        # Checking Login Status
        if not self.ensure_login():
            return "login_failed"

        # Construct base URL (adds usr and pwd in the call)
        base_url = (
            f"{self.BASE_URL}?action=adjustments&symbol={symbol}"
            f"&startdate={self.config.start_date}&enddate={self.config.end_date}"
        )
        # Add credentials to URL
        url = self.add_auth_to_url(base_url)
            
        for attempt in range(max_attempts):
            response, status = self.http_client.request(url)
            
            if status != "success":
                if attempt < max_attempts - 1:
                    # Exponential wait between attempts
                    wait_time = 2 ** attempt
                    logger.info(f"Tentativo {attempt+1}/{max_attempts} fallito per aggiustamento {symbol}. "
                               f"Attesa di {wait_time}s prima del prossimo tentativo.")
                    time.sleep(wait_time)
                continue  # Try again
            
            # Analyses the answer
            response_text = response.text
            
            # Check if we have no data
            if "405 Data Not Found" in response_text[:100]:
                return "no_data"
            
            # Check whether the session has expired
            if "401 Not Logged In" in response_text:
                self.is_logged_in = False
                # Re-login and try one last time
                if self._login():
                    return self.download_adjustment_data(symbol, max_attempts=1)
                return "session_expired"
            
            # Success - save data to S3
            try:
                df = pd.read_csv(BytesIO(response.content), sep='\t')
                
                if df.empty:
                    return "empty_data"
                
                # Save data to S3 instead of local disk
                if self.s3_processor:
                    success = self.s3_processor.save_adjustment_data_to_s3(df, symbol, self.config.asset_type)
                    return "success" if success else "save_failed"
                else:
                    logger.error("S3 processor not available for saving adjustment data")
                    return "s3_processor_missing"
                
            except Exception as parse_error:
                logger.error(f"Error during analysis of adjustment data for {symbol}: {parse_error}")
                return f"parse_error: {str(parse_error)}"
        
        # If we get here, all attempts have failed
        return "max_retries_reached"

class DataProcessor:
    """Processes downloaded data easily and efficiently"""
    
    # Configuration of columns for different data types
    COLUMN_CONFIGS = {
        'tickms': {'time': 1, 'price': 2, 'volume': 3},
        'tick': {'time': 1, 'price': 2, 'volume': 3},
        'tickbidaskms': {'time': 1, 'price': 2, 'bid': 3, 'ask': 4, 'size': 5},
        'tickbidask': {'time': 1, 'price': 2, 'bid': 3, 'ask': 4, 'size': 5}
    }
    
    @staticmethod
    def process_data(df, data_type, asset_type, chunk_size=None, retry_on_error=True, **kwargs):
        """
        Processes data efficiently with error handling
        """
        if df is None or df.empty:
            logger.warning("DataFrame empty or None, cannot process")
            return None
        
        try:
                       
            # If the data type is not supported, return only the relevant columns
            if data_type not in DataProcessor.COLUMN_CONFIGS:
                logger.info(f"Data type '{data_type}' not recognised, return base columns")
                return df.iloc[:, 1:]
            
            # Extract the configuration for this data type
            cols = DataProcessor.COLUMN_CONFIGS[data_type]
            time_col = cols['time']
           

            # Calculates the trade count (number of trades per timestamp)
            df['trades'] = df.groupby(time_col)[time_col].transform('count')

            # Sort by timestamp
            df = df.sort_values(by=time_col)
            
             # Processes according to asset and data type
            if asset_type and asset_type.lower() in ['forex', 'futures']:
                # For forex and futures we aggregate by holding the 'last' for price, bid and ask
                result = DataProcessor._process_forex_futures_data(df, cols)
            else:
                # For stocks, etfs and other types of assets we continue with the existing logic
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
                    # Fallback: tentativo di elaborazione semplificata
                    return DataProcessor._simple_process(df, data_type)
                except Exception as fallback_error:
                    logger.error(f"The fallback processing also failed: {str(fallback_error)}")
                    
            # In case of a complete error, returns at least the basic columns
            return df.iloc[:, 1:]
    
    @staticmethod
    def _simple_process(df, data_type):
        """Simplified processing as fallback in case of errors"""
        cols = DataProcessor.COLUMN_CONFIGS.get(data_type, {})
        if not cols:
            return df.iloc[:, 1:]
            
        time_col = cols.get('time', 1)
        
        # Simple aggregation by timestamp
        df = df.sort_values(by=time_col)
        return df
    
    @staticmethod
    def _process_tick_data(df, cols):
        """Processes tick data specifically"""
        time_col = cols['time']
        price_col = cols['price']
        volume_col = cols['volume']
        
        # Calculate weighted average for each timestamp
        df['weight'] = df[volume_col] / df.groupby(time_col)[volume_col].transform('sum')
        df['weighted_price'] = df[price_col] * df['weight']

        # Aggregate by timestamp
        agg_df = df.groupby(time_col).agg({
            'weighted_price': 'sum',
            volume_col: 'sum',
            'trades': 'last'
        }).reset_index()

        # Rename columns
        agg_df.rename(columns={'weighted_price': price_col}, inplace=True)
        return agg_df
    
    @staticmethod
    def _process_bidask_data(df, cols):
        """Processes bid/ask data specifically"""
        time_col = cols['time']
        price_col = cols['price']
        size_col = cols['size']
        bid_col = cols['bid']
        ask_col = cols['ask']
        
        # Calculates the weighted average for each timestamp
        df['weight'] = df[size_col] / df.groupby(time_col)[size_col].transform('sum')
        # Calculates price-weighted bid and ask values
        df['weighted_price'] = df[price_col] * df['weight']
        df['weighted_bid'] = df[bid_col] * df['weight']
        df['weighted_ask'] = df[ask_col] * df['weight']
        
        # Aggregate by timestamp with standard columns
        agg_columns = {
            'weighted_price': 'sum',
            'weighted_bid': 'sum',
            'weighted_ask': 'sum',
            size_col: 'sum',
            'trades': 'last'
        }
        
        # Perform main aggregation
        agg_df = df.groupby(time_col).agg(agg_columns).reset_index()
        
        # Rename weighted columns to their original names
        agg_df.rename(columns={
            'weighted_price': price_col,
            'weighted_bid': bid_col,
            'weighted_ask': ask_col
        }, inplace=True)
        
        return agg_df
    
    @staticmethod
    def _process_forex_futures_data(df, cols):
        """
        It processes forex or futures data, aggregating by timestamps
        and taking the most recent value for price, bid and ask
        """
        time_col = cols['time']
        price_col = cols['price']
        bid_col = cols['bid']
        ask_col = cols['ask']
        volume_col = cols['size']

        # For forex/futures, we take the median price of each group
        agg_columns = {
            price_col: 'median',
            bid_col: 'median',
            ask_col: 'median',
            volume_col: 'sum',
            'trades': 'last'
        }
        
        # Perform aggregation
        agg_df = df.groupby(time_col).agg(agg_columns).reset_index()
        
        return agg_df

    @staticmethod
    def save_data(df, file_path, file_format='parquet', max_retries=3, retry_delay=2):
        """
        Save data to disk safely and efficiently with retry mechanism
        
        Args:
            df: DataFrame to be saved
            file_path: File path without extension
            file_format: Saving format ('txt' or 'parquet')
            max_retries: Maximum number of retries in case of error
            retry_delay: Waiting time between retries (seconds, with exponential backoff)
            
        Returns:
            True if the save was successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"Attempt to save empty DataFrame in {file_path}")
            return False
        
        # Create the directory
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Unable to create directory {directory}: {str(e)}")
            return False
        
        # Determine the extent and final route
        if file_format == 'parquet':
            extension = '.parquet'
        else:
            extension = '.txt'
            
        final_path = f"{file_path}{extension}"       
                
        # Multiple rescue attempts
        for attempt in range(max_retries):
            # Generates a unique name for the temporary file
            temp_path = f"{file_path}.tmp.{attempt}{extension}"
            
            try:
                logger.debug(f"Tentativo {attempt+1}/{max_retries} di salvataggio in {final_path}")
                
                # Save data with format-optimised parameters
                if file_format == 'parquet':
                    # Import pyarrow only if necessary
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    
                    df.columns = [str(col) for col in df.columns]
                    # Save in parquet format
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, temp_path, compression='snappy')
                else:
                    # Save in CSV/TXT format
                    df.to_csv(
                        temp_path,
                        float_format='%.15g',  # 15 significant digits (max for float64)
                        date_format='%Y-%m-%d %H:%M:%S.%f',  # Microsecond precision
                        index=False,  # Include index
                        header=False,  # Include headers 
                        chunksize=50000 if len(df) > 100000 else None
                    )
                
                # Check that the file has been saved correctly
                if not os.path.exists(temp_path):
                    raise IOError(f"File temporaneo non creato: {temp_path}")
                    
                if os.path.getsize(temp_path) == 0:
                    raise IOError(f"File temporaneo vuoto: {temp_path}")
                
                # Checks file integrity
                try:
                    # Basic verification by reading the
                    with open(temp_path, 'rb') as f:
                        # Read the first 1000 bytes for basic verification
                        header = f.read(1000)
                        if len(header) < 10:
                            raise IOError("File created but too small")
                except Exception as integrity_error:
                    logger.warning(f"Integrity check failed: {str(integrity_error)}")
                    raise IOError(f"Temporary file not readable: {temp_path}")
                
                # Atomically rename
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)
                
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                
                # Clean up temporary files
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # If not the last attempt, wait and try again
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting for {wait_time}s before the next attempt")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All save attempts failed for {file_path}")
        
        return False

    @staticmethod
    def verify_file(file_path, file_format='parquet'):
        """
        Verify the integrity of a saved file
        """
        extension = '.parquet' if file_format == 'parquet' else '.txt'
        full_path = f"{file_path}{extension}"
        
        if not os.path.exists(full_path):
            return False, f"File not found: {full_path}"
        
        if os.path.getsize(full_path) == 0:
            return False, f"Empty file: {full_path}"
        
        try:
            if file_format == 'parquet':
                # Import pyarrow only if necessary
                import pyarrow.parquet as pq
                
                # Check that it is a valid parquet file by reading the metadata
                metadata = pq.read_metadata(full_path)
                if metadata.num_rows == 0:
                    return False, f"Readable but empty parquet file: {full_path}"
            else:
                # CSV/TXT file verification
                df = pd.read_csv(full_path, nrows=1)
                if df is None or df.empty:
                    return False, f"Readable but empty file: {full_path}"
                    
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
            
            # Log ogni step%
            if percent // self.log_step > self.last_logged_percent // self.log_step:
                logger.info(f"Progress: {percent}% completed ({self.completed_items}/{self.total_items})")
                self.last_logged_percent = percent
                return True
        return False

class KibotDownloader:
    """Optimised downloader for Kibot data with advanced error handling and integrity checks"""
    def __init__(self, config):
        self.config = config
        
        # Initialize S3 processor for MinIO storage
        self.s3_processor = S3DataProcessor(config)
        
        # Unified HTTP client with integrated rate limiter
        self.http_client = HTTPClient(
            max_connections= min(config.max_concurrent_connections, 4),  # Max. 4 simultaneous connections as per Kibot guide
            user_agent="KibotPythonClient",
            connection_timeout=config.connection_timeout
        )
        
        # API client with S3 processor reference
        self.api = KibotAPI(config, self.http_client, self.s3_processor)
        
        # Tracing existing files
        self.existing_files = {}
        
        # Extensive statistics
        self.stats = {
            "total": 0,
            "downloaded": 0,
            "existing": 0,
            "no_data": 0,
            "failed": 0,
            "failed_downloads": 0,
            "failed_processing": 0,
            "processed": 0,
            "verified": 0,
            "invalid": 0,
            "timeout": 0,
            "retried": 0,
            "retry_success": 0
        }
        self.lock = threading.RLock()
        
        # Extended Separate Processing Queue
        self.processing_queue = queue.Queue(maxsize=100)
        
        # Queue for failed attempts
        self.retry_queue = queue.Queue()
        
        # Termination flag
        self.processing_done = threading.Event()
        
        # Registry of processed files to avoid reprocessing
        self.processed_registry = set()
        
       # Flag to pause downloads when memory is critical
        self.pause_downloads = False
        
        # Counter for garbage collection
        self.processed_count = 0
        self.gc_frequency = 50
        
        # Tracking downloads by symbol/date for diagnostics
        self.item_status = defaultdict(dict)
        
    def check_memory_critically(self):
        """More efficient memory control"""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 90:  # Only for real emergencies
            # Download break and cleaning
            self.pause_downloads = True
            gc.collect(generation=2)
            self.api.response_cache.clear()
            
            # Wait a maximum of 5 seconds
            for _ in range(5):
                gc.collect()
                if psutil.virtual_memory().percent < 85:
                    break
                time.sleep(1)
                
            self.pause_downloads = False
        elif memory_percent > 80:  # Light cleaning
            gc.collect()

    def log_download_info(self):
        """Log download information - SAVE TO S3 instead of local files"""
        log_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = datetime.strptime(self.config.start_date, "%m/%d/%Y").strftime("%Y_%m_%d")
        end_date = datetime.strptime(self.config.end_date, "%m/%d/%Y").strftime("%Y_%m_%d")
        
        for symbol in self.config.symbols:
            log_info = f"{log_date}, {start_date}, {end_date}\n"
            # Save to S3 instead of local file
            self.s3_processor.save_last_update_info_to_s3(symbol, self.config.asset_type, log_info)
    
    def collect_existing_files(self):
        """Gathers information on existing files to optimise verification - CHECK S3 instead of local"""
        logger.info("Gathering information on existing files in S3...")
        
        for symbol in self.config.symbols:
            # For S3, we'll check files on-demand rather than pre-loading
            # This is more efficient for S3 storage
            self.existing_files[symbol] = set()
            logger.debug(f"Initialized S3 file tracking for {symbol}")

    def is_file_existing(self, symbol, date):
        """Optimised check if a file already exists - CHECK S3 instead of local"""
        # Registry check (already processed in this session)
        item_key = f"{symbol}_{date}"
        if item_key in self.processed_registry:
            return True
            
        # Dictionary check in memory (cache)
        date_filename = PathManager.get_date_filename(date)
        if symbol in self.existing_files and date_filename in self.existing_files[symbol]:
            return True
        
        # S3 verification instead of local disk
        exists = self.s3_processor.file_exists_in_s3(symbol, date, self.config.asset_type, self.config.file_format)
        
        # Refresh the cache if the file exists
        if exists:
            with self.lock:
                if symbol not in self.existing_files:
                    self.existing_files[symbol] = set()
                self.existing_files[symbol].add(date_filename)
        
        return exists

    def generate_dates(self):
        """Generate dates for which to download data"""
        start_dt = datetime.strptime(self.config.start_date, "%m/%d/%Y")
        end_dt = datetime.strptime(self.config.end_date, "%m/%d/%Y")
        
        # Excludes weekends for equities, not for forex/futures
        exclude_we = self.config.asset_type not in ('forex', 'futures')
        
        if exclude_we:
            dates = pd.bdate_range(start_dt, end_dt)
        else:
            dates = pd.date_range(start_dt, end_dt)
        
        return [d.strftime("%m/%d/%Y") for d in dates] 
    
    def generate_work_items(self):
        """Generate work items by avoiding existing files"""
        dates = self.generate_dates()
        
        # Collect information on existing files
        self.collect_existing_files()
        
        # Pre-calculates the total to show an estimate
        total_potential = len(self.config.symbols) * len(dates)
        logger.info(f"Analisi di {total_potential} potenziali file da scaricare...")
        
        # Generate elements more efficiently
        work_items = []
        for symbol in self.config.symbols:
            missing_items = [
                (symbol, date) for date in dates
                if not self.is_file_existing(symbol, date)
            ]
            work_items.extend(missing_items)
            
            if missing_items:
                logger.debug(f"Found {len(missing_items)} missing files for {symbol}")
        
        # Shuffle only if necessary for load balancing
        if len(work_items) > 100:  # Avoid shuffles for small sets
            random.shuffle(work_items)
        
        logger.info(f"Found {len(work_items)} missing files to download for {len({symbol for symbol, _ in work_items})} symbols")
        return work_items

    def _track_item_status(self, symbol, date, status):
        """Track the status of an element by diagnostics"""
        item_key = f"{symbol}_{date}"
        with self.lock:
            if status not in self.item_status[item_key]:
                self.item_status[item_key][status] = 0
            self.item_status[item_key][status] += 1
    
    def download_item(self, item):
        """Downloading a single item and adding it to the processing queue"""
        symbol, date = item
        
        # Unique key for this file
        item_key = f"{symbol}_{date}"
        
        # Check whether it has already been successfully processed
        if item_key in self.processed_registry:
            return "already_processed"
        
        # Quick check if the file already exists
        if self.is_file_existing(symbol, date):
            with self.lock:
                self.stats["existing"] += 1
                self.stats["total"] += 1
            self._track_item_status(symbol, date, "existing")
            return "existing"
        
        # Download data
        df, status = self.api.download_data(symbol, date)
      
        self._track_item_status(symbol, date, status)

        # If the download fails but is critical (auth_failed), try again immediately
        if status == "auth_failed":
            logger.warning(f"Authentication failure for {symbol} {date}, immediate recovery attempt")
            
            # Resetting the login in the API class
            self.api.is_logged_in = False
            
            # Force new login
            if self.api._login():
                # Try the download again
                df, status = self.api.download_data(symbol, date)
                
                if status != "success":
                    # If it still fails, add to the high priority retry queue
                    self.retry_queue.put((symbol, date))
                    logger.error(f"Critical failure for {symbol} {date} after re-login. Added to retry queue.")
            else:
                logger.error(f"Unable to re-login for {symbol} {date}")
                self.retry_queue.put((symbol, date))
        
        # Update Progress Monitor
        self.progress_monitor.update()
        
        # Update Statistics
        with self.lock:
            self.stats["total"] += 1
            if status == "success":
                self.stats["downloaded"] += 1
            elif status == "no_data":
                self.stats["no_data"] += 1
            elif status == "timeout_error":
                self.stats["timeout"] += 1
                self.stats["failed"] += 1
                self.stats["failed_downloads"] += 1
            elif not status.startswith("cached"):
                self.stats["failed"] += 1
                self.stats["failed_downloads"] += 1
        
        # If the download was successful, add to the processing queue
        if status == "success" or status == "cached":
            try:
                # Add to queue with timeout to avoid blockages
                self.processing_queue.put((symbol, date, df, 0), block=True, timeout=10)
                return "queued_for_processing"
            except queue.Full:
                logger.warning(f"Full processing queue. Immediate processing for {symbol} {date}")
                # If the queue is full, it processes directly
                success = self._process_item(symbol, date, df)
                return "processed_immediately" if success else "processing_failed"
        
        return status

    def _process_item(self, symbol, date, df, attempt=0, max_attempts=3):
        """Elaboration of a simplified element - SAVE TO S3 instead of local"""
        # Checking previous processing
        item_key = f"{symbol}_{date}"
        if item_key in self.processed_registry:
            return True
        
        try:
            # Process data
            processed_df = (
                DataProcessor.process_data(df, self.config.frequency, self.config.asset_type)
                if self.config.should_process_data
                else df.iloc[:, 1:]
            )
            
            # DataFrame Verification
            if processed_df is None or processed_df.empty:
                raise ValueError("DataFrame processed is empty")
            
            # Save data to S3 instead of local disk
            success = self.s3_processor.save_data_to_s3(
                processed_df, symbol, date, self.config.asset_type, self.config.file_format
            ) 
            
            if success:
                # Update registers
                self.processed_registry.add(item_key)
                with self.lock:
                    self.stats["processed"] += 1
                    date_filename = PathManager.get_date_filename(date)
                    self.existing_files.setdefault(symbol, set()).add(date_filename)
                
                return True
                
        except Exception as e:
            logger.error(f"Processing error {symbol} {date}: {e}")
            
        # Simplified bankruptcy handling
        if attempt < max_attempts - 1:
            return self._process_item(symbol, date, df, attempt + 1, max_attempts)
        
        with self.lock:
            self.stats["failed"] += 1
            self.stats["failed_processing"] += 1
        
        return False
  
    def process_retry_queue(self):
        """Handles elements that failed during processing"""
        if self.retry_queue.empty():
            logger.info("No items in the retry queue")
            return
        
        retry_items = []
        while not self.retry_queue.empty():
            try:
                item = self.retry_queue.get_nowait()
                retry_items.append(item)
                self.retry_queue.task_done()
            except queue.Empty:
                break
        
        # Separate elements by type of failure
        auth_failed_items = []
        other_failed_items = []
        
        for item in retry_items:
            symbol, date = item
            item_key = f"{symbol}_{date}"
            
            if 'auth_failed' in self.item_status.get(item_key, {}):
                auth_failed_items.append(item)
            else:
                other_failed_items.append(item)
        
        # Process the elements
        successful = 0
        for items, description in [(auth_failed_items, "autenticazione"), (other_failed_items, "altri")]:
            if items:
                logger.info(f"Processing {len(items)} elements with {description} errors")
                for symbol, date in items:
                    try:
                        df, status = self.api.download_data(symbol, date)
                        self._track_item_status(symbol, date, f"retry_{status}")
                        
                        if status == "success" or status == "cached":
                            success = self._process_item(symbol, date, df)
                            if success:
                                successful += 1
                                with self.lock:
                                    self.stats["retry_success"] += 1
                    except Exception as e:
                        logger.error(f"Error during retry for {symbol} {date}: {e}")
        
        logger.info(f"Completed retry: {successful}/{len(retry_items)} processed successfully")

    def process_item_from_queue(self):
        """Worker for processing elements from the queue"""
        while not self.processing_done.is_set():
            try:
                # Takes an element from the queue with timeout
                try:
                    symbol, date, df, attempts = self.processing_queue.get(timeout=1)
                except queue.Empty:
                    # The queue is empty, but may receive other elements
                    continue
                
                try:
                    # Process the element
                    max_attempts = 3  # Max. processing attempts
                    if attempts < max_attempts:
                        success = self._process_item(symbol, date, df, attempts, max_attempts)
                        
                        # If it fails, increase attempts and put back in the queue
                        if not success and attempts < max_attempts - 1:
                            self.processing_queue.put((symbol, date, df, attempts + 1))
                            with self.lock:
                                self.stats["retried"] += 1
                    else:
                        logger.warning(f"Too many failed processing attempts for {symbol} {date}")
                        # Add to retry queue if configured
                        if self.config.enable_retry_queue:
                            self.retry_queue.put((symbol, date))
                            self._track_item_status(symbol, date, "retry_queued")
                
                except Exception as e:
                    logger.error(f"Error in worker during data processing for {symbol} {date}: {e}", exc_info=True)
                
                finally:
                    # Memory cleaning after processing
                    try:
                        del df
                    except:
                        pass
                    
                    # Report that the item was processed
                    self.processing_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Error in the processing worker: {e}", exc_info=True)
                
    def run(self):
        """Performs full data download with optimised resource management"""
        start_time = time.time()
        
        try:
            # For S3 storage, we don't need to create local directories
            # S3 handles path creation automatically
            
            # Shows the current configuration in the log
            logger.info(f"Start downloading for symbols={self.config.symbols}, " + 
                       f"start date={self.config.start_date}, " +
                       f"end date={self.config.end_date}, " )
            logger.info(f"Configuration: max_concurrent_connections={self.config.max_concurrent_connections}, " + 
                       f"max_download_workers={self.config.max_download_workers}, " +
                       f"max_processing_workers={self.config.max_processing_workers}, " +
                       f"timeout={self.config.connection_timeout}s")
            
            # Register download information
            self.log_download_info()

            # Calculates the total number of potential files
            dates = self.generate_dates()
            total_potential = len(self.config.symbols) * len(dates)

            # Generate work elements
            work_items = self.generate_work_items()
            total_items = len(work_items)

            # Calculates the number of existing files and updates statistics
            existing_files = total_potential - total_items
            with self.lock:
                self.stats["existing"] = existing_files
                self.stats["total"] += existing_files

            # Initialise the progress monitor
            self.progress_monitor = ProgressMonitor(total_items, log_step=10)
                    
            if total_items == 0:
                logger.info("No new data to download. All files are already existing.")
                # download the adjustment data if required
                if self.config.download_adjustments:
                    self.download_adjustments()
                
                return True
            
            logger.info(f"Downloading {total_items} file for {len(self.config.symbols)} symbols...")
            
            # Start processing workers
            self.processing_done.clear()
            processing_workers = []
            for i in range(self.config.max_processing_workers):
                worker = threading.Thread(
                    target=self.process_item_from_queue,
                    name=f"ProcessingWorker-{i}"
                )
                worker.daemon = True
                worker.start()
                processing_workers.append(worker)
            
            logger.info(f"Avviati {len(processing_workers)} worker di processing")
            
            # Dynamic chunk size based on data type and total number
            chunk_size = min(500, total_items)
            
            # Run parallel download with memory control
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.config.max_download_workers, 4)) as executor:
                for i in range(0, total_items, chunk_size):
                    # Check if we need to take a break for memory's sake
                    while self.pause_downloads:
                        time.sleep(0.5)
                        self.check_memory_critically()
                    
                    chunk = work_items[i:i+chunk_size]
                    
                    # Download the chunk
                    futures = [executor.submit(self.download_item, item) for item in chunk]
                    concurrent.futures.wait(futures)
                    
                    # Memory check every chunk
                    self.check_memory_critically()
                    
                    # Wait for the processing queue to reduce if it is too full
                    while self.processing_queue.qsize() > 50:  # reduced from 500
                        logger.info(f"Processing queue full ({self.processing_queue.qsize()}), awaiting processing...")
                        time.sleep(2)
                        self.check_memory_critically()
            
            # Wait for the processing queue to be empty
            logger.info("Download complete. Waiting for processing to complete...")
            
            # Monitoring processing completion
            while self.processing_queue.qsize() > 0:
                remaining = self.processing_queue.qsize()
                logger.info(f"Remaining files to process: {remaining}")
                time.sleep(5)
            
            # Wait for final completion
            self.processing_queue.join()
            
            # Process retry queue if enabled and not empty
            if self.config.enable_retry_queue and not self.retry_queue.empty():
                logger.info("Processing failed elements...")
                self.process_retry_queue()
            
            # Signals processing workers to terminate
            self.processing_done.set()
            
            # Wait for all processing workers to finish
            for worker in processing_workers:
                worker.join(timeout=5)
            
            # Download adjustment data if required
            if self.config.download_adjustments:
                logger.info("Start downloading adjustment data...")
                self.download_adjustments()
                logger.info("Download adjustment data completed")
            
            # Calcola statistiche
            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
            
            # Stampa riepilogo
            summary = (
                "====== DOWNLOAD SUMMARY ======\n"
                f"Completed in: {total_time_str}\n"
                f"Total files: {self.stats['total']}\n"
                f"- Downloaded: {self.stats['downloaded']}\n"
                f"- Pre-existing: {self.stats['existing']}\n"
                f"- Without data: {self.stats['no_data']}\n"
                f"- Failed: {self.stats['failed']}\n"
                f"- Failed downloads: {self.stats['failed_downloads']}\n"
                f"- Failed processing: {self.stats['failed_processing']}\n"
                f"- Timeout: {self.stats['timeout']}\n"
                f"- Retried files: {self.stats['retried']}\n"
                f"- Successful recoveries: {self.stats['retry_success']}\n"
                f"File processed: {self.stats['processed']}\n"
                f"Average speed: {self.stats['downloaded'] / max(1, total_time):.2f} files/s\n"
                f"CPU usage: {psutil.cpu_percent()}%\n"
                f"Memory usage: {psutil.virtual_memory().percent}%\n"
                "=============================="
            )
            logger.info(summary)
            print(summary)
            
            # Generate detailed report - CONSOLE OUTPUT INSTEAD OF FILE
            if self.config.generate_detailed_report:
                self._generate_detailed_report()
            
            return True
            
        except Exception as e:
            logger.error(f"Errore durante il download: {e}", exc_info=True)
            return False
        finally:
            # Signals processing workers to terminate
            self.processing_done.set()
            # Close all connections
            self.http_client.close_all()
    
    def _generate_detailed_report(self):
        """Generate a detailed report of the operation with symbol analysis and discrepancy detection - CONSOLE OUTPUT"""
        try:
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
            
            # CONSOLE OUTPUT INSTEAD OF FILE
            print("\n===== DETAILED REPORT =====")
            
            # Statistiche generali
            print("\nGENERAL STATISTICS")
            for key, value in self.stats.items():
                print(f"{key}: {value}")
            
            # Statistiche per simbolo
            print("\n===== STATISTICS BY SYMBOL =====")
            for symbol in sorted(symbol_stats.keys()):
                stats = symbol_stats[symbol]
                print(f"\n{symbol}:")
                print(f"  success: {stats['success']}")
                print(f"  no_data: {stats['no_data']}")
                print(f"  total: {stats['total']}")
            
            # Controllo discrepanze tra simboli
            print("\n===== DISCREPANCY ANALYSIS =====")
            
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
                        print(f"\n WARNING: Discrepanze tra {reference_symbol} e {symbol}:")
                        
                        if success_only_in_ref:
                            print(f"  SUCCESS solo in {reference_symbol}: {sorted(success_only_in_ref)}")
                        if success_only_in_curr:
                            print(f"  SUCCESS solo in {symbol}: {sorted(success_only_in_curr)}")
                        if no_data_only_in_ref:
                            print(f"  NO_DATA solo in {reference_symbol}: {sorted(no_data_only_in_ref)}")
                        if no_data_only_in_curr:
                            print(f"  NO_DATA solo in {symbol}: {sorted(no_data_only_in_curr)}")
                
                if not discrepancies_found:
                    print(" Nessuna discrepanza trovata tra i simboli")
            else:
                print("Un solo simbolo presente - controllo discrepanze non applicabile")
            
            # Dettaglio status per elemento (sezione originale)
            print("\n===== ELEMENT STATUS DETAIL =====")
            for item_key, status_dict in self.item_status.items():
                print(f"{item_key}: {status_dict}")
                        
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def download_adjustments(self):
        """Download adjustment data for all symbols with improved error handling"""
        logger.info(f"Download adjustment data for {len(self.config.symbols)} symbols")
        results = {}
        failed_symbols = []
        
        # Limit connections to 4 as recommended by Kibot
        max_workers = min(self.config.max_download_workers, 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create the futures
            futures = {
                executor.submit(self.api.download_adjustment_data, symbol): symbol
                for symbol in self.config.symbols
            }
            
            # Process results as they come in
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    logger.info(f"Adjustment {symbol}: {result}")
                    
                    # Check your memory periodically
                    self.check_memory_critically()
                    
                except Exception as e:
                    error_msg = f"Error downloading adjustment for {symbol}: {e}"
                    logger.error(error_msg)
                    results[symbol] = f"error: {str(e)}"
                    failed_symbols.append(symbol)
        
        # Retry for failed symbols
        if failed_symbols and self.config.retry_failed_adjustments:
            logger.info(f"Attempt to re-download fixes for {len(failed_symbols)} symbols failed")
            
            for symbol in failed_symbols:
                try:
                    # Short wait to avoid rate limit problems
                    time.sleep(1)
                    result = self.api.download_adjustment_data(symbol, max_attempts=2)
                    results[symbol] = result
                    logger.info(f"Retry adjustment {symbol}: {result}")
                except Exception as e:
                    logger.error(f"The retry also failed for {symbol}: {e}")
        
        # Summary of adjustment results
        success_count = sum(1 for result in results.values() if result == "success")
        logger.info(f"Download adjustments completed: {success_count}/{len(self.config.symbols)} riusciti")
        
        return results

# Configuration class
class Config:
    """Kibot Downloader Setup with Advanced Parameters"""
    def __init__(self, 
                 user, 
                 pwd, 
                 base_dir, 
                 symbols,
                 start_date,
                 end_date,
                 asset_type="stocks",
                 frequency= None,
                 max_concurrent_connections=4,  # Limited to 4 as per Kibot recommendations
                 max_download_workers=4,
                 max_processing_workers=4,
                 connection_timeout=60,
                 data_timeout=120,
                 download_adjustments=None,
                 should_process_data=None,
                 file_format="parquet", # or parquet
                 chunk_size=50000,
                 retry_count=4,
                 # New advanced parameters
                 verify_file_integrity=True,
                 enable_retry_queue=True,
                 retry_failed_adjustments=True,
                 generate_detailed_report=True):
        # Login parameters
        self.user = user
        self.pwd = pwd
        
        # Data parameters
        self.base_dir = base_dir
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.asset_type = asset_type

        # AUTOMATIC CONFIGURATIONS BASED ON ASSET_TYPE
        if asset_type in ['stocks', 'ETFs']:
            self.frequency = frequency or "tickms"
            self.download_adjustments = download_adjustments if download_adjustments is not None else True
            self.should_process_data = should_process_data if should_process_data is not None else True
        elif asset_type in ['forex', 'futures']:
            self.frequency = frequency or "tickbidaskms"
            self.download_adjustments = download_adjustments if download_adjustments is not None else False
            self.should_process_data = should_process_data if should_process_data is not None else True
        else:
            # Default configuration
            self.frequency = frequency or "tickms"
            self.download_adjustments = download_adjustments if download_adjustments is not None else True
            self.should_process_data = should_process_data if should_process_data is not None else True
            
        # Connection parameters
        self.max_concurrent_connections = min(max_concurrent_connections, 4)  # Max 4 connessioni come da guida Kibot
        self.max_download_workers = min(max_download_workers, self.max_concurrent_connections)
        self.max_processing_workers = max_processing_workers
        self.connection_timeout = connection_timeout
        self.data_timeout = data_timeout
        
        # Processing parameters
        self.file_format = file_format.lower()
        if self.file_format not in ['txt', 'parquet']:
            logger.warning(f"File format '{file_format}' not supported. Using 'txt' as default.")
            self.file_format = 'txt'
        self.chunk_size = chunk_size
        self.retry_count = retry_count
        
        # Advanced parameters
        self.verify_file_integrity = verify_file_integrity
        self.enable_retry_queue = enable_retry_queue
        self.retry_failed_adjustments = retry_failed_adjustments
        self.generate_detailed_report = generate_detailed_report
        
        # Set appropriate datatypes based on frequency
        self.datatype = {
            0: np.str_,  # Data
            1: np.str_,  # Time
            2: np.float64  # Price
        }
        
        if self.frequency.startswith('tick'):
            if 'bidask' in self.frequency:
                # For bid/ask tick data
                self.datatype.update({
                    3: np.float64,  # Bid
                    4: np.float64,  # Ask
                    5: np.int64     # Size
                })
            else:
                # For normal tick data
                self.datatype.update({
                    3: np.int64     # Volume
                })
    
    def __str__(self):
        """Provides a readable representation of the configuration"""
        return (
            f"Config(user='{self.user}', "
            f"base_dir='{self.base_dir}', "
            f"symbols=[{len(self.symbols)} simboli], "
            f"periodo={self.start_date}-{self.end_date}, "
            f"frequency='{self.frequency}', "
            f"asset_type='{self.asset_type}', "
            f"max_connections={self.max_concurrent_connections}, "
            f"file_format={self.file_format})"
        )

# ============================================================
# MAIN FUNCTION FOR WINDMILL
# ============================================================

def main(
    # Pipeline configuration
    pipeline_name="download_data",
    pipeline_enabled=True,
    
    # Asset type configuration - IDENTICAL to config.json structure
    stocks_batch1_enabled=True,
    stocks_batch1_symbols=["MDT", "AAPL", "ADBE", "AMD", "AMZN", "AXP", "BA", "CAT", "COIN", "CSCO", "DIS", "EBAY", "GE", "GOOGL", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MSFT", "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "PYPL", "SHOP", "SNAP", "SPOT", "TSLA", "UBER", "V", "WMT", "XOM", "ZM", "ABBV", "ABT", "ACN", "AIG", "AMGN", "AMT", "AVGO", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CVS", "CVX", "DE", "DHR", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM", "GOOG", "HON", "INTU", "KHC", "LIN", "LLY", "LMT", "LOW", "MA", "MDLZ", "MET", "MO", "MRK", "MS", "NEE", "PEP", "PFE", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "VZ", "WFC"],
    stocks_batch1_start_date="03/01/2024",
    stocks_batch1_end_date="03/01/2025",
    
    stocks_batch2_enabled=False,
    stocks_batch2_symbols=["TEST"],
    stocks_batch2_start_date="03/01/2024",
    stocks_batch2_end_date="03/01/2025",
    
    etfs_enabled=False,
    etfs_symbols=["AGG", "BND", "GLD", "SLV", "SUSA", "EFIV", "ESGV", "ESGU", "AFTY", "MCHI", "EWH", "EEM", "IEUR", "VGK", "FLCH", "EWJ", "NKY", "EWZ", "EWC", "EWU", "EWI", "EWP", "ACWI", "IOO", "GWL", "VEU", "IJH", "MDY", "IVOO", "IYT", "XTN", "XLI", "XLU", "VPU", "SPSM", "IJR", "VIOO", "QQQ", "ICLN", "ARKK", "SPLG", "SPY", "VOO", "IYY", "VTI", "DIA"],
    etfs_start_date="03/01/2024",
    etfs_end_date="03/01/2025",
    
    forex_enabled=False,
    forex_symbols=["EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "JPYUSD", "CHFUSD", "SGDUSD", "HKDUSD", "KRWUSD", "INRUSD", "RUBUSD", "BRLUSD"],
    forex_start_date="03/01/2024",
    forex_end_date="03/01/2025",
    
    futures_enabled=False,
    futures_symbols=["ES", "CL", "GC", "NG", "NQ", "TY", "FV", "EU", "SI", "C", "W", "VX"],
    futures_start_date="03/01/2024",
    futures_end_date="03/01/2025",
    
    # General settings - IDENTICAL to config.json
    base_dir=None,  # Will use Windmill variable
    file_format="parquet",
    execution_mode="async",
    system_cores_reserved=1,
    download_threads_reserved=4,
    processing_threads_reserved=4,
    max_concurrent_connections=4,
    connection_timeout=60,
    data_timeout=120,
    verify_file_integrity=True,
    enable_retry_queue=True,
    retry_failed_adjustments=True,
    generate_detailed_report=True
):
    """
    Windmill-Compatible Kibot Data Download
    
    Preserves IDENTICAL logic to original download.py but with:
    - Console logging instead of file logging
    - Windmill variable integration
    - Slack notifications
    - Parameter structure matching config.json
    """
    start_time = datetime.now()
    
    # Setup console logging
    print(f"KIBOT DATA DOWNLOAD START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Get Windmill variables
        kibot_user = wmill.get_variable("u/niccolosalvini27/KIBOT_USER")
        kibot_password = wmill.get_variable("u/niccolosalvini27/KIBOT_PASSWORD")
        if base_dir is None:
            base_dir = wmill.get_variable("u/niccolosalvini27/BASE_DIR")
        
        if not kibot_user or not kibot_password or not base_dir:
            raise ValueError("Missing required Windmill variables: KIBOT_USER, KIBOT_PASSWORD, BASE_DIR")
        
        # Build enabled pipelines list - IDENTICAL to original structure
        enabled_pipelines = []
        
        if stocks_batch1_enabled:
            enabled_pipelines.append({
                'name': 'stocks_batch1',
                'asset_type': 'stocks',
                'symbols': stocks_batch1_symbols,
                'start_date': stocks_batch1_start_date,
                'end_date': stocks_batch1_end_date,
                'steps': ['download']
            })
        
        if stocks_batch2_enabled:
            enabled_pipelines.append({
                'name': 'stocks_batch2',
                'asset_type': 'stocks',
                'symbols': stocks_batch2_symbols,
                'start_date': stocks_batch2_start_date,
                'end_date': stocks_batch2_end_date,
                'steps': ['download']
            })
        
        if etfs_enabled:
            enabled_pipelines.append({
                'name': 'ETFs',
                'asset_type': 'ETFs',
                'symbols': etfs_symbols,
                'start_date': etfs_start_date,
                'end_date': etfs_end_date,
                'steps': ['download']
            })
        
        if forex_enabled:
            enabled_pipelines.append({
                'name': 'forex',
                'asset_type': 'forex',
                'symbols': forex_symbols,
                'start_date': forex_start_date,
                'end_date': forex_end_date,
                'steps': ['download']
            })
        
        if futures_enabled:
            enabled_pipelines.append({
                'name': 'futures',
                'asset_type': 'futures',
                'symbols': futures_symbols,
                'start_date': futures_start_date,
                'end_date': futures_end_date,
                'steps': ['download']
            })
        
        if not enabled_pipelines:
            print("No pipelines enabled. Exiting.")
            return
        
        print(f"Enabled pipelines: {[p['name'] for p in enabled_pipelines]}")
        
        # Process each pipeline - IDENTICAL to original logic
        all_results = []
        total_processed = 0
        total_errors = 0
        total_files = 0
        total_downloaded = 0
        total_existing = 0
        
        for pipeline in enabled_pipelines:
            print(f"\n--- Processing pipeline: {pipeline['name']} ({pipeline['asset_type']}) ---")
            
            # Create Config object - IDENTICAL to original
            config = Config(
                user=kibot_user,
                pwd=kibot_password,
                base_dir=base_dir,
                symbols=pipeline['symbols'],
                start_date=pipeline['start_date'],
                end_date=pipeline['end_date'],
                asset_type=pipeline['asset_type'],
                max_concurrent_connections=max_concurrent_connections,
                max_download_workers=download_threads_reserved,
                max_processing_workers=processing_threads_reserved,
                connection_timeout=connection_timeout,
                data_timeout=data_timeout,
                file_format=file_format,
                verify_file_integrity=verify_file_integrity,
                enable_retry_queue=enable_retry_queue,
                retry_failed_adjustments=retry_failed_adjustments,
                generate_detailed_report=generate_detailed_report
            )
            
            print(f"Configuration: {config}")
            
            # Create and run downloader - IDENTICAL logic
            downloader = KibotDownloader(config)
            success = downloader.run()
            
            # Collect results
            pipeline_result = {
                'pipeline': pipeline['name'],
                'asset_type': pipeline['asset_type'],
                'success': success,
                'stats': downloader.stats.copy()
            }
            all_results.append(pipeline_result)
            
            # Update totals
            total_processed += downloader.stats['processed']
            total_errors += downloader.stats['failed']
            total_files += downloader.stats['total']
            total_downloaded += downloader.stats['downloaded']
            total_existing += downloader.stats['existing']
            
            print(f"Pipeline {pipeline['name']} completed: {'SUCCESS' if success else 'FAILED'}")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        successful_pipelines = [r for r in all_results if r['success']]
        success_rate = len(successful_pipelines) / len(all_results) * 100 if all_results else 0
        
        print("\n====== FINAL DOWNLOAD SUMMARY ======")
        print(f"Total execution time: {duration}")
        print(f"Pipelines processed: {len(all_results)}")
        print(f"Successful pipelines: {len(successful_pipelines)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total files: {total_files}")
        print(f"Downloaded: {total_downloaded}")
        print(f"Pre-existing: {total_existing}")
        print(f"Processed: {total_processed}")
        print(f"Errors: {total_errors}")
        
        # Determine status
        if total_errors == 0 and len(successful_pipelines) == len(all_results):
            print("ALL DOWNLOAD PROCESSING COMPLETED SUCCESSFULLY!")
            status_emoji = "✅"
            status_text = "SUCCESS"
        else:
            print("SOME DOWNLOAD PROCESSING ERRORS OCCURRED!")
            status_emoji = "⚠️" if len(successful_pipelines) > 0 else "❌"
            status_text = "PARTIAL SUCCESS" if len(successful_pipelines) > 0 else "FAILURE"
        
        # Create Slack message
        slack_message = f"{status_emoji} *KIBOT DATA DOWNLOAD COMPLETED* {status_emoji}\n\n"
        slack_message += f"*Status:* {status_text}\n"
        slack_message += f"*Pipeline:* {pipeline_name}\n"
        slack_message += f"*Duration:* {str(duration).split('.')[0]}\n"
        slack_message += f"*Pipelines:* {len(successful_pipelines)}/{len(all_results)} successful\n\n"
        
        # Add pipeline details
        slack_message += "*Pipeline Results:*\n"
        for result in all_results:
            status_icon = "✅" if result['success'] else "❌"
            slack_message += f"{status_icon} {result['pipeline']} ({result['asset_type']}): {result['stats']['downloaded']} downloaded, {result['stats']['failed']} errors\n"
        
        slack_message += f"\n*Processing Summary:*"
        slack_message += f"\n📁 Total files: {total_files}"
        slack_message += f"\n⬇️ Downloaded: {total_downloaded}"
        slack_message += f"\n📋 Pre-existing: {total_existing}"
        slack_message += f"\n✅ Processed: {total_processed}"
        slack_message += f"\n❌ Errors: {total_errors}"
        
        if total_files > 0:
            slack_message += f"\n📊 Success rate: {success_rate:.1f}%"
        
        slack_message += f"\n\n*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
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
            error_message = f"❌ *KIBOT DATA DOWNLOAD FAILED* ❌\n\n"
            error_message += f"*Error:* {str(e)}\n"
            error_message += f"*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            send_slack_notification(error_message)
        except Exception as slack_error:
            print(f"Failed to send error notification: {slack_error}")
