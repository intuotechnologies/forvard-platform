"""
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

2025 University of Messina, Department of Economics.
Research code - Unauthorized distribution prohibited.
"""

from rv_library import*
import time
import re
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
from datetime import datetime
import numpy as np
import threading

# Thread-safe lock per l'accesso al file CSV
_file_lock = threading.Lock()

def setup_logging(config):
    """Setup logging configuration with both console and file output"""
    log_dir = os.path.join(config['base_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"variance_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger('volatility_processor')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger, log_filepath

def get_start_date_from_log(log_file_path, asset_type):
    """Extract start date from log file"""
    try:
        if asset_type not in log_file_path:
            symbol = os.path.basename(log_file_path).replace('_last_update.txt', '')
            base_dir = os.path.dirname(os.path.dirname(log_file_path))
            log_file_path = os.path.join(base_dir, asset_type, symbol, f'{symbol}_last_update.txt')
        
        if not os.path.exists(log_file_path):
            return None

        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            lines = log_file.readlines()
            
            if not lines:
                return None
                
            last_line = lines[-1].strip()
            if not last_line:
                return None
                
            parts = last_line.split(',')
            if len(parts) < 2:
                return None
                
            second_column = parts[1].strip()
            try:
                start_date = datetime.strptime(second_column, "%Y_%m_%d").strftime("%Y_%m_%d")
                return start_date
            except ValueError:
                return None
                
    except Exception:
        return None

def validate_result_data(result):
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
                     'minrv1', 'minrv5', 'minrv5_ss', 'rk']
    
    for field in numeric_fields:
        if field in result and result[field] is not None:
            if np.isnan(result[field]) or np.isinf(result[field]):
                return False
    
    return True

def process_single_file_with_retry(file_path, file_date, symbol, config, max_retries=2):
    """Process a single raw data file with retry mechanism"""
    logger = logging.getLogger('volatility_processor')
    
    for attempt in range(max_retries + 1):
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return None
            
            df = prepare_data_rv(file_path, config)
            if df is None or df.empty:
                return None
           
            # Extract OHLC values
            if config['asset_type'] in ['futures', 'forex']:
                num_trades = df['trades'].sum() if 'trades' in df.columns else 0
                volume = 0
            else:
                num_trades = df['trades'].sum() if 'trades' in df.columns else 0
                volume = df['volume'].sum() if 'volume' in df.columns else 0

            # Calculate volatility measures with error handling
            volatility_results = {}
            
            try:
                volatility_results['rv1'] = realized_power_variation(df, exp = 2, resample_freq='1T', price_col='price', 
                              resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                rv5_result = realized_power_variation(df, exp = 2, resample_freq='5T', price_col='price', 
                              resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['rv5'] = rv5_result[0] if isinstance(rv5_result, tuple) else rv5_result
                volatility_results['rv5_ss'] = rv5_result[1] if isinstance(rv5_result, tuple) else None
            except Exception:
                volatility_results.update({'rv1': None, 'rv5': None, 'rv5_ss': None})
            # Quarticity
            try:
                volatility_results['rq1'] = realized_power_variation(df, exp= 4, resample_freq='1T', price_col='price', 
                              resampling_method='last', calculate_subsample=False, n_subsamples=5)
           
                rq5_result = realized_power_variation(df, exp=4, resample_freq='5T', price_col='price', 
                              resampling_method='last', calculate_subsample=True, n_subsamples=5)
                
                volatility_results['rq5'] = rq5_result[0] if isinstance(rq5_result, tuple) else rq5_result
                volatility_results['rq5_ss'] = rq5_result[1] if isinstance(rq5_result, tuple) else None
               
            except Exception:
                volatility_results.update({'rq1': None, 'rq5': None, 'rq5_ss': None})
            
            try:
                volatility_results['bv1'] = bipower_variation(df, resample_freq='1T', price_col='price', 
                              resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                bv5_result = bipower_variation(df, resample_freq='5T', price_col='price', 
                              resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['bv5'] = bv5_result[0] if isinstance(bv5_result, tuple) else bv5_result
                volatility_results['bv5_ss'] = bv5_result[1] if isinstance(bv5_result, tuple) else None
            except Exception:
                volatility_results.update({'bv1': None, 'bv5': None, 'bv5_ss': None})
            
            try:
                rs_result_1 = realized_semivariance(df, resample_freq='1T', price_col='price', 
                                 resampling_method='last', calculate_subsample=False, n_subsamples=5)
                volatility_results['rsp1'] = rs_result_1[0] if isinstance(rs_result_1, tuple) else None
                volatility_results['rsn1'] = rs_result_1[1] if isinstance(rs_result_1, tuple) else None
                
                rs_result_5 = realized_semivariance(df, resample_freq='5T', price_col='price', 
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
                volatility_results['medrv1'] = median_realized_variance(df, resample_freq='1T', price_col='price', 
                                    resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                medrv5_result = median_realized_variance(df, resample_freq='5T', price_col='price', 
                                    resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['medrv5'] = medrv5_result[0] if isinstance(medrv5_result, tuple) else medrv5_result
                volatility_results['medrv5_ss'] = medrv5_result[1] if isinstance(medrv5_result, tuple) else None
            except Exception:
                volatility_results.update({'medrv1': None, 'medrv5': None, 'medrv5_ss': None})
            
            try:
                volatility_results['minrv1'] = min_realized_variance(df, resample_freq='1T', price_col='price', 
                                    resampling_method='last', calculate_subsample=False, n_subsamples=5)
                
                minrv5_result = min_realized_variance(df, resample_freq='5T', price_col='price', 
                                    resampling_method='last', calculate_subsample=True, n_subsamples=5)
                volatility_results['minrv5'] = minrv5_result[0] if isinstance(minrv5_result, tuple) else minrv5_result
                volatility_results['minrv5_ss'] = minrv5_result[1] if isinstance(minrv5_result, tuple) else None
            except Exception:
                volatility_results.update({'minrv1': None, 'minrv5': None, 'minrv5_ss': None})

            try:
                volatility_results['rk'] = realized_kernel_variance(df, resample_freq='1S', price_col='price', resampling_method='last')
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
                logger.warning(f"Attempt {attempt + 1} failed for {os.path.basename(file_path)}: {e}. Retrying...")
                time.sleep(1)
            else:
                logger.error(f"All attempts failed for {os.path.basename(file_path)}: {e}")
                return None

def update_results_batch(results_batch, output_dir):
    """Update the central results file with a batch of results - Simple and efficient version"""
    logger = logging.getLogger('volatility_processor')
    
    if not results_batch:
        return 0, 0
    
    # Validate results
    valid_results = [r for r in results_batch if validate_result_data(r)]
    if not valid_results:
        logger.error("No valid results in batch")
        return 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "realized_variance.csv")
    
    with _file_lock:
        try:
            # Read existing data
            existing_df = pd.DataFrame()
            if os.path.exists(output_file):
                try:
                    existing_df = pd.read_csv(output_file, sep=',', encoding='utf-8')
                except Exception as e:
                    logger.error(f"Error reading existing file: {e}")
                    # Backup corrupted file
                    backup_file = f"{output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    try:
                        os.rename(output_file, backup_file)
                        logger.warning(f"Corrupted file backed up to {backup_file}")
                    except Exception:
                        pass
                    existing_df = pd.DataFrame()
            
            new_df = pd.DataFrame(valid_results)
            
            # Handle duplicates
            if not existing_df.empty:
                existing_keys = set(zip(existing_df['symbol'].astype(str), existing_df['date'].astype(str), existing_df['asset_type'].astype(str)))

                unique_results = []
                skipped_count = 0
                
                for result in valid_results:
                    key = (str(result['symbol']), str(result['date']), str(result['asset_type']))
                    if key not in existing_keys:
                        unique_results.append(result)
                    else:
                        skipped_count += 1
            
                if unique_results:
                    new_df = pd.DataFrame(unique_results)
                    new_df = new_df.dropna(axis=1, how='all')
                    existing_df_clean = existing_df.dropna(axis=1, how='all')
                    combined_df = pd.concat([existing_df_clean, new_df], ignore_index=True)
                    combined_df = combined_df.sort_values(['symbol', 'date'], ignore_index=True)
                    combined_df.to_csv(output_file, sep=',', index=False, encoding='utf-8')
                    return skipped_count, len(unique_results)
                else:
                    return len(valid_results), 0
            else:
                new_df = new_df.sort_values(['symbol', 'date'], ignore_index=True)
                new_df.to_csv(output_file, sep=',', index=False, encoding='utf-8')
                return 0, len(valid_results)
                
        except Exception as e:
            logger.error(f"Critical error in update_results_batch: {e}")
            return 0, 0

def check_missing_dates(symbol, valid_files, output_dir):
    """Check which dates are missing from the output file"""
    try:
        expected_dates = set()
        for file_path, date_match in valid_files:
            try:
                file_date = datetime.strptime(date_match.group(1), '%Y_%m_%d')
                expected_dates.add(file_date.strftime('%Y-%m-%d'))
            except Exception:
                continue
        
        processed_dates = set()
        output_file = os.path.join(output_dir, "realized_variance.csv")
        
        if os.path.exists(output_file):
            try:
                df = pd.read_csv(output_file, sep=',', encoding='utf-8')
                if not df.empty:
                    symbol_df = df[df['symbol'].astype(str) == str(symbol)]
                    processed_dates = set(symbol_df['date'].astype(str).tolist())
            except Exception:
                pass
        
        return expected_dates - processed_dates
        
    except Exception:
        return set()

def process_files_in_batches_with_retry(file_batches, symbol, config, output_dir):
    """Process files in batches with retry for failed files"""
    logger = logging.getLogger('volatility_processor')
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    failed_files = []
    
    for batch_idx, batch in enumerate(file_batches):
        results_batch = []
        for file_path, date_match in batch:
            try:
                file_date = datetime.strptime(date_match.group(1), '%Y_%m_%d')
                result = process_single_file_with_retry(file_path, file_date, symbol, config)
                
                if result:
                    results_batch.append(result)
                else:
                    total_errors += 1
                    failed_files.append((file_path, date_match))
                    
            except Exception as e:
                logger.error(f"Error processing {os.path.basename(file_path)}: {e}")
                total_errors += 1
                failed_files.append((file_path, date_match))
        
        if results_batch:
            try:
                skipped, processed = update_results_batch(results_batch, output_dir)
                total_skipped += skipped
                total_processed += processed
            except Exception as e:
                logger.error(f"Error updating results for batch {batch_idx + 1}: {e}")
                total_errors += len(results_batch)
    
    return total_processed, total_skipped, total_errors, failed_files

def process_symbol_files_with_verification(symbol, config, output_dir):
    """Process all files for a single symbol with verification and retry"""
    logger = logging.getLogger('volatility_processor')
    
    start_time = time.time()
    
    try:
        asset_type = config['asset_type']
        symbol_dir = os.path.join(config['base_dir'], asset_type, symbol)

        if not os.path.exists(symbol_dir):
            logger.warning(f"Directory for {symbol} does not exist: {symbol_dir}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "missing_dates": 0, "total_files": 0}

        log_file_path = os.path.join(symbol_dir, f"{symbol}_last_update.txt")
        start_date = get_start_date_from_log(log_file_path, asset_type)

        if start_date:
            logger.info(f"Processing {symbol} from start date: {start_date}")
        
        file_format = config.get('file_format', 'txt').lower()
        file_extension = '.parquet' if file_format == 'parquet' else '.txt'
        excluded_files = ['adjustment.txt', f'{symbol}_last_update.txt'] if file_format == 'txt' else []
        
        all_files = []
        try:
            for filename in os.listdir(symbol_dir):
                if not filename.endswith(file_extension) or filename in excluded_files:
                    continue
                    
                if start_date:
                    file_date_part = filename.replace(file_extension, '')
                    if file_date_part < start_date:
                        continue
                
                all_files.append(filename)
        except Exception as e:
            logger.error(f"Error listing files in {symbol_dir}: {e}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, "missing_dates": 0, "total_files": 0}
        
        files = sorted(all_files)
        
        if not files:
            logger.warning(f"No files to process for {symbol}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "missing_dates": 0, "total_files": 0}
        
        valid_files = []
        date_pattern = r'(\d{4}_\d{2}_\d{2})' + re.escape(file_extension)
        
        for file in files:
            date_match = re.search(date_pattern, file)
            if date_match:
                valid_files.append((os.path.join(symbol_dir, file), date_match))
        
        if not valid_files:
            logger.warning(f"No valid date-formatted files for {symbol}")
            return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 0, "missing_dates": 0, "total_files": 0}
        
        # Pre-check existing records to avoid unnecessary processing
        output_file = os.path.join(output_dir, "realized_variance.csv")
        existing_dates = set()
        
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file, sep=',', encoding='utf-8')
                if not existing_df.empty:
                    #symbol_df = existing_df[existing_df['symbol'].astype(str) == str(symbol)]
                    symbol_df = existing_df[(existing_df['symbol'].astype(str) == str(symbol)) & 
                       (existing_df['asset_type'].astype(str) == str(config['asset_type']))]
                    existing_dates = set(symbol_df['date'].astype(str).tolist())
            except Exception as e:
                logger.warning(f"Could not read existing CSV for pre-check: {e}")
                pass
        
        # Filter out files that already exist
        files_to_process = []
        for file_path, date_match in valid_files:
            try:
                file_date = datetime.strptime(date_match.group(1), '%Y_%m_%d')
                date_str = file_date.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    files_to_process.append((file_path, date_match))
            except Exception:
                # If date parsing fails, include the file anyway
                files_to_process.append((file_path, date_match))
        
        total_files = len(valid_files)
        files_to_process_count = len(files_to_process)
        already_processed = total_files - files_to_process_count
        
        if already_processed > 0:
            logger.info(f"Processing {symbol}: {files_to_process_count} new files, {already_processed} already processed, {total_files} total files")
        else:
            logger.info(f"Processing {symbol}: {files_to_process_count} files (all new)")
        
        if not files_to_process:
            logger.info(f"{symbol}: All files already processed")
            return {"symbol": symbol, "processed": 0, "skipped": already_processed, "errors": 0, 
                   "missing_dates": 0, "total_files": total_files, "processing_time": 0}
        
        # Simple batch size determination
        batch_size = min(20, max(5, files_to_process_count // 10))
        file_batches = [files_to_process[i:i+batch_size] for i in range(0, len(files_to_process), batch_size)]
        
        processed, skipped, errors, failed_files = process_files_in_batches_with_retry(
            file_batches, symbol, config, output_dir)
        
        # Add the pre-skipped files to total skipped count
        skipped += already_processed
        
        missing_dates = check_missing_dates(symbol, valid_files, output_dir)
        
        # Retry failed files once
        if failed_files and len(failed_files) <= 10:
            logger.info(f"Retrying {len(failed_files)} failed files for {symbol}")
            retry_processed, retry_skipped, retry_errors, _ = process_files_in_batches_with_retry(
                [failed_files], symbol, config, output_dir)
            processed += retry_processed
            errors = retry_errors
        
        processing_time = time.time() - start_time
        
        #if errors == 0 and len(missing_dates) == 0:
        #    logger.info(f"{symbol} completed successfully in {processing_time:.1f}s: {processed} processed, {skipped} skipped")
        #else:
        #    logger.warning(f"{symbol} completed with issues in {processing_time:.1f}s: {processed} processed, {skipped} skipped, {errors} errors, {len(missing_dates)} missing")
        
        return {
            "symbol": symbol, 
            "processed": processed, 
            "skipped": skipped, 
            "errors": errors, 
            "missing_dates": len(missing_dates),
            "total_files": total_files,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Critical error processing {symbol}: {e}")
        return {"symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, 
                "missing_dates": 0, "total_files": 0, "processing_time": 0}

def process_all_symbols_threads_with_verification(config):
    """Process all symbols using thread-based parallelism - Simple and efficient"""
    logger = logging.getLogger('volatility_processor')
    
    if not config.get('symbols') or not os.path.exists(config.get('base_dir', '')):
        logger.error("Invalid configuration")
        return []
    
    output_dir = config['base_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    num_symbols = len(config['symbols'])
    max_workers = config.get('max_workers', min(max(1, os.cpu_count() - 1), num_symbols, 4))
    
    logger.info("Starting volatility processing")
    logger.info(f"Workers: {max_workers}, Symbols: {num_symbols}, Asset type: {config['asset_type']}")
    
    start_time = time.time()
    results = []
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(process_symbol_files_with_verification, symbol, config, output_dir): symbol 
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
                    #logger.info(f"Progress: {completed}/{num_symbols} ({progress:.0f}%) - {symbol}: {result['processed']} processed, {result['errors']} errors")
                    # Complete log with all information
                    status = "SUCCESS" if result['errors'] == 0 and result.get('missing_dates', 0) == 0 else "WARNING"
                    logger.info(f"Progress: {completed}/{num_symbols} ({progress:.0f}%) - "
                            f"COMPLETED: {symbol} ({status}) - "
                            f"{result['processed']} processed, {result['skipped']} skipped, "
                            f"{result['errors']} errors in {result.get('processing_time', 0):.1f}s")

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results.append({
                        "symbol": symbol, "processed": 0, "skipped": 0, "errors": 1, 
                        "missing_dates": 0, "total_files": 0, "processing_time": 0
                    })
                    completed += 1
    
    except Exception as e:
        logger.error(f"Critical error in thread execution: {e}")
        return results
    
    # Summary
    total_elapsed = time.time() - start_time
    total_processed = sum(r["processed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    #total_missing = sum(r.get("missing_dates", 0) for r in results)
    total_files = sum(r.get("total_files", 0) for r in results)
    
    successful_symbols = [r for r in results if r['errors'] == 0 and r['missing_dates'] == 0]
    symbols_with_errors = [r for r in results if r["errors"] > 0]
    symbols_with_missing = [r for r in results if r.get("missing_dates", 0) > 0]
    
    logger.info("Processing completed")
    logger.info(f"Total time: {total_elapsed:.1f}s, Files: {total_files}, Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")
    logger.info(f"Symbols: {len(results)} total, {len(successful_symbols)} successful, {len(symbols_with_errors)} with errors, {len(symbols_with_missing)} with missing dates")
    
    if symbols_with_errors:
        logger.warning("Symbols with errors:")
        for r in symbols_with_errors:
            logger.warning(f"  {r['symbol']}: {r['errors']} errors")
    
    if symbols_with_missing:
        logger.warning("Symbols with missing dates:")
        for r in symbols_with_missing:
            logger.warning(f"  {r['symbol']}: {r['missing_dates']} missing dates")
    
    # Final file check
    output_file = os.path.join(output_dir, "realized_variance.csv")
    if os.path.exists(output_file):
        try:
            final_df = pd.read_csv(output_file, sep=',', encoding='utf-8')
            logger.info(f"Output file: {len(final_df)} records, {final_df['symbol'].nunique()} symbols, {os.path.getsize(output_file)/(1024*1024):.1f} MB")
        except Exception:
            logger.warning("Could not verify output file")
    
    return results