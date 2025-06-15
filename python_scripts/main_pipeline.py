"""
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)
2025 University of Messina, Department of Economics.

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 26/05/2025

Research code - Unauthorized distribution prohibited.
"""

import json
import sys
import argparse
import threading
import queue
from queue import Empty
import os
import time
import psutil
import logging
import concurrent.futures
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Importing external modules
from download import KibotDownloader, Config as DownloadConfig
from outliers_detection import process_all_symbols_threaded as process_outliers
from compute_rv import process_all_symbols_threads_with_verification as process_rv

# Global synchronisation structures
print_lock = threading.Lock()
current_pipeline_lock = threading.Lock()
download_results_lock = threading.Lock()

# Semahore to limit the number of concurrent downloads (Kibot limit)
kibot_connection_semaphore = threading.BoundedSemaphore(4)

# Thread-safe container for asynchronous download results
download_results_container = {}

# Global download log to prevent duplicates between pipelines
downloaded_symbols_registry = {}
downloaded_symbols_lock = threading.Lock()

# Event to indicate when processing can start
processing_can_start = threading.Event()

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
        else:
            print(message)

def load_symbols(symbols_file):
    """Load symbols from file"""
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            return symbols
    except FileNotFoundError:
        safe_print(f"Symbol file not found: {symbols_file}", "ERROR")
        return []

def load_config(config_file="config.json"):
    """Load configuration from JSON file and symbols from external files"""
    
    # Load environment variables from the .env file
    load_dotenv()
    
    # Read the configuration file
    with open(config_file, 'r', encoding='utf-8') as f:
        config_text = f.read()
    
    # Take environment variables and convert backslash to forward slash
    kibot_user = os.getenv('KIBOT_USER', '')
    kibot_password = os.getenv('KIBOT_PASSWORD', '')
    base_dir = os.getenv('BASE_DIR', '').replace('\\', '/')
    config_dir = os.getenv('CONFIG_DIR', '').replace('\\', '/')
    
    # Replace variables
    config_text = config_text.replace('${KIBOT_USER}', kibot_user)
    config_text = config_text.replace('${KIBOT_PASSWORD}', kibot_password)
    config_text = config_text.replace('${BASE_DIR}', base_dir)
    config_text = config_text.replace('${CONFIG_DIR}', config_dir)
    
    # JSON Parse
    config = json.loads(config_text)
    
    # Load symbols from external files for enabled pipelines
    for pipeline_name, pipeline_config in config['pipelines'].items():
        if pipeline_config.get('enabled', True):
            if 'symbols_file' in pipeline_config:
                symbols_file = pipeline_config['symbols_file']
                pipeline_config['symbols'] = load_symbols(symbols_file)
                if not pipeline_config['symbols']:
                    safe_print(f"ATTENTION: No symbol found for {pipeline_name}", "WARNING")
                    pipeline_config['enabled'] = False
        else:
            safe_print(f"Disabled {pipeline_name} Pipeline", "INFO")
            
    return config


def clean_kibot_logger():
    """Cleans handlers of the KibotDownloader logger to avoid duplicates"""
    kibot_logger = logging.getLogger("KibotDownloader")
    
    # If the logger has no handler, do nothing
    if not kibot_logger.handlers:
        return
    
    # Remove all handlers except the first one
    while len(kibot_logger.handlers) > 1:
        handler = kibot_logger.handlers[-1]
        kibot_logger.removeHandler(handler)
        try:
            handler.close()
        except:
            pass
    
    # Set propagate=False to avoid duplicates to the root logger
    kibot_logger.propagate = False

def calculate_optimal_workers(config, num_symbols, operation_type):
    """Calculates the optimal number of workers for various operations"""
    cpu_count = os.cpu_count() or 4
    
    # Get configuration parameters
    system_reserved = config['general'].get('system_cores_reserved', 1)
    download_reserved = config['general'].get('download_threads_reserved', 4)
    processing_reserved = config['general'].get('processing_threads_reserved', 2)
    
    # Calculate available cores
    available_cores = cpu_count - system_reserved
    
    # Determines the cores to be used according to the operation
    if operation_type.lower() == 'outliers':
        # Consider whether there are active downloads
        downloads_active = kibot_connection_semaphore._value < 4
        if downloads_active:
            # Download in progress, reserve resources
            available_cores -= (download_reserved + processing_reserved)
        
        # Add specific limits for outliers
        max_outliers = config['general'].get('outliers_threads_max', available_cores)
        optimal_workers = min(available_cores, max_outliers, num_symbols)
    
    elif operation_type.lower() == 'rv':
        # Consider whether there are active downloads
        downloads_active = kibot_connection_semaphore._value < 4
        if downloads_active:
            available_cores -= (download_reserved + processing_reserved)
        
        # Add RV-specific limits
        max_rv = config['general'].get('rv_threads_max', available_cores)
        optimal_workers = min(available_cores, max_rv, num_symbols)
    
    else:  # download
        # limit to 4 per download (Kibot limit)
        optimal_workers = min(4, num_symbols)
    
    # Make sure you have at least 1 worker
    optimal_workers = max(1, optimal_workers)
    
    safe_print(f"{operation_type.upper()}: Allocated {optimal_workers} worker(s) for {num_symbols} symbols", "WORKERS")
    return optimal_workers

def is_symbol_already_downloaded(symbol, date, asset_type):
    """
    Check in the global register whether a symbol has already been downloaded or is in progress
    """
    with downloaded_symbols_lock:
        key = f"{symbol}_{date}_{asset_type}"
        return key in downloaded_symbols_registry

def mark_symbol_as_downloaded(symbol, date, asset_type, status=True):
    """
    Mark a symbol as downloaded in the global registry
    """
    with downloaded_symbols_lock:
        key = f"{symbol}_{date}_{asset_type}"
        downloaded_symbols_registry[key] = status

def download_worker(config, pipeline_name, symbols_batch, result_queue):
    '''Worker function for batch data download with global register check'''
    # List of symbols actually to be downloaded (after duplicate filter)
    symbols_to_download = []
    asset_type = config['pipelines'][pipeline_name]['asset_type']
    date_range = config['pipelines'][pipeline_name]['date_range']
    
    # Filter symbols already downloaded/processed
    for symbol in symbols_batch:
        # Check if the symbol is already in the global register for all dates
        # we only use the start and end date
        start_date = date_range['start_date']
        end_date = date_range['end_date']
        
        if (not is_symbol_already_downloaded(symbol, start_date, asset_type) and 
            not is_symbol_already_downloaded(symbol, end_date, asset_type)):
            symbols_to_download.append(symbol)
            # Mark in advance as being downloaded
            mark_symbol_as_downloaded(symbol, start_date, asset_type)
            mark_symbol_as_downloaded(symbol, end_date, asset_type)
    
    # If all symbols are already downloaded, it ends immediately
    if not symbols_to_download:
        safe_print(f"No new symbols to download for {pipeline_name}, all already processed", "INFO")
        result_queue.put({
            'pipeline': pipeline_name,
            'symbols': symbols_batch,  # Return the original symbols for processing
            'status': True,
            'asset_type': asset_type
        })
        return True
    
    # Proceed with downloading only for the necessary symbols
    safe_print(f"Batch download of {len(symbols_to_download)}/{len(symbols_batch)} symbols for {pipeline_name}", "DOWNLOAD")
    
    # Acquires the semaphore to limit total connections to Kibot
    with kibot_connection_semaphore:
        try:
            # Clean KibotDownloader logger to avoid duplicates
            clean_kibot_logger()
            
            # Set up the downloader with optimal parameters
            download_config = DownloadConfig(
                user=config['credentials']['kibot_user'],
                pwd=config['credentials']['kibot_password'],
                base_dir=config['general']['base_dir'],
                symbols=symbols_to_download,  # Use filtered symbols only
                start_date=date_range['start_date'],
                end_date=date_range['end_date'],
                asset_type=asset_type,
                file_format=config['general']['file_format'],
                max_concurrent_connections=4,
                max_download_workers=4,
                max_processing_workers=config['general'].get('processing_threads_reserved', 2)
            )
            
            downloader = KibotDownloader(download_config)
            
            # Run the download
            success = downloader.run()
            
            # Put the result in the results queue
            result_queue.put({
                'pipeline': pipeline_name,
                'symbols': symbols_batch,  # Return the original symbols for processing
                'status': success,
                'asset_type': asset_type
            })
            
            safe_print(f"Download completed batch of {len(symbols_to_download)} symbols for {pipeline_name}", "COMPLETE")
            return success
            
        except Exception as e:
            safe_print(f"Batch download error of {pipeline_name}: {e}", "ERROR")
            # In the event of an error, remove the symbols from the logClean KibotDownloader logger to avoid duplicates
            for symbol in symbols_to_download:
                mark_symbol_as_downloaded(symbol, date_range['start_date'], asset_type, False)
                mark_symbol_as_downloaded(symbol, date_range['end_date'], asset_type, False)
            return False

def process_pipeline(config, pipeline_name, batches, steps_to_run):
    """Processes a complete pipeline (outliers and realised variance)"""
    try:
        safe_print(f"Start processing pipeline {pipeline_name}", "PROCESS")
        
        # Merge all batches of symbols
        all_symbols = []
        asset_type = None
        
        for batch in batches:
            all_symbols.extend(batch['symbols'])
            # Get the asset type from the first batch
            if asset_type is None:
                asset_type = batch['asset_type']
        
        # Remove duplicates while maintaining order
        all_symbols = list(dict.fromkeys(all_symbols))
        
        safe_print(f"Processing {len(all_symbols)} symbols for {pipeline_name}", "PROCESS")
        
        # Process outliers if the asset type is stocks or etf and required in the steps
        if asset_type.lower() in ['stocks', 'ETFs'] and 'outliers' in steps_to_run:
            safe_print(f"Outliers detection for {len(all_symbols)} symbols in {pipeline_name}", "STEP")
            
            # Calculate the optimal number of workers per outliers
            optimal_workers = calculate_optimal_workers(config, len(all_symbols), 'outliers')
            
            # Set up and start outliers
            outliers_config = {
                'base_dir': config['general']['base_dir'],
                'file_format': config['general']['file_format'],
                'asset_type': asset_type,
                'symbols': all_symbols,
                'early_closing_day_file': config['general']['early_closing_day_file'],
                'max_workers': optimal_workers}            
            
            start_time = time.time()
            process_outliers(outliers_config)
            end_time = time.time()
            
            safe_print(f"Outliers completed  in {end_time - start_time:.2f}s", "COMPLETE")
        
        # Processa realized variance se richiesto nei passi
        if 'realized_variance' in steps_to_run:
            safe_print(f"Realised variance calculation for {len(all_symbols)} symbols in {pipeline_name}", "STEP")
            
            # Calculate the optimal number of RV workers
            optimal_workers = calculate_optimal_workers(config, len(all_symbols), 'rv')
            
            # Import the logger and configure RV
            from compute_rv import setup_logging
            
            rv_config = {
                'base_dir': config['general']['base_dir'],
                'symbols': all_symbols,
                'asset_type': asset_type,
                'file_format': config['general']['file_format'],
                'max_workers': optimal_workers                
            }
            
            # Initialise the logger for realised variance
            logger, log_filepath = setup_logging(rv_config)
            logger.info(f"Starting realized variance processing in {pipeline_name}")
            
            start_time = time.time()
            process_rv(rv_config)
            end_time = time.time()
            
            safe_print(f"Realized variance completed in {end_time - start_time:.2f}s", "COMPLETE")
        
        safe_print(f"Processing pipeline {pipeline_name} completed", "COMPLETE")
        return True
        
    except Exception as e:
        safe_print(f"Pipeline {pipeline_name} processing error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

def run_pipeline_download(config, pipeline_name, selected_steps=None):
    """Executes only the download phase of the pipeline and returns downloaded batches"""
    safe_print(f"Pipeline {pipeline_name.upper()}", "PHASE")
    
    pipeline_config = config['pipelines'][pipeline_name]
    
    # Only proceed if the download is in the steps to be performed
    steps_to_run = selected_steps if selected_steps else pipeline_config['steps']
    
    safe_print(f"Steps to be executed: {steps_to_run}")
    safe_print(f"Symbols: {len(pipeline_config['symbols'])}, Type: {pipeline_config['asset_type']}")
    
    # Clean the KibotDownloader logger before you start
    clean_kibot_logger()
    
    # Batch symbols for parallel processing
    all_symbols = pipeline_config['symbols']
    
    # Create batches of appropriate size
    if len(all_symbols) <= 10:
        # With few symbols, put them all in one batch
        batch_size = len(all_symbols)
    elif len(all_symbols) < 20:
        batch_size = max(5, len(all_symbols) // 2)
    elif len(all_symbols) < 50:
        batch_size = max(10, len(all_symbols) // 3)
    else:
        batch_size = max(15, len(all_symbols) // (os.cpu_count() or 4))
    
    safe_print(f"Batch size: {batch_size} simboli")
    
    # Split symbols in batches
    symbol_batches = [all_symbols[i:i+batch_size] for i in range(0, len(all_symbols), batch_size)]
    
    # If there is no download, create a dummy batch and return it
    if 'download' not in steps_to_run:
        safe_print(f"Nessun download richiesto per {pipeline_name}, solo processing", "INFO")
        return [{
            'pipeline': pipeline_name,
            'symbols': all_symbols,
            'status': True,
            'asset_type': pipeline_config['asset_type']
        }], steps_to_run
    
    # Create a queue for download results
    result_queue = queue.Queue()
    
    # Create a download executor for this pipeline
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submitting all batches for downloading
        futures = []
        for batch in symbol_batches:
            future = executor.submit(download_worker, config, pipeline_name, batch, result_queue)
            futures.append(future)
        
        # Wait for all downloads to complete
        success = True
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_success = future.result()
                if not batch_success:
                    safe_print(f"Batch download failed for{pipeline_name}", "ERROR")
                    success = False
            except Exception as e:
                safe_print(f"Exception in the batch download of {pipeline_name}: {e}", "ERROR")
                success = False
    
    # Collect all downloaded batches from the queue
    downloaded_batches = []
    while not result_queue.empty():
        try:
            batch = result_queue.get_nowait()
            downloaded_batches.append(batch)
            result_queue.task_done()
        except Empty:
            break
    
    safe_print(f"Download pipeline {pipeline_name} completed with {len(downloaded_batches)} batch", "COMPLETE")
    return downloaded_batches, steps_to_run

def safe_download_next_pipeline(config, pipeline_name, steps):
    """Safe function to download the next pipeline in a separate thread"""
    try:
        safe_print(f"Parallel download initialisation of {pipeline_name}", "DOWNLOAD")
        
        # Run the download
        result = run_pipeline_download(config, pipeline_name, steps)
        
        # Save the result in the shared container
        with download_results_lock:
            download_results_container[pipeline_name] = result
            
        safe_print(f"Parallel download of {pipeline_name} completed and saved in the container", "COMPLETE")
        
        # Notification that download is complete
        processing_can_start.set()
        
    except Exception as e:
        safe_print(f"Error in parallel download of {pipeline_name}: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        
        # Save an empty result in case of error
        with download_results_lock:
            download_results_container[pipeline_name] = (None, None)
        
        # Notification even in case of error
        processing_can_start.set()

# optimized 1:32:13.263480
def execute_pipeline_async(config, pipelines_to_run, selected_steps=None):
    """
    Optimised implementation to maximise machine utilisation:
    - Downloading (4 threads, I/O bound)
    - Processing (remaining cores, CPU bound)
    - Real overlap between download and processing of different pipelines
    """
    # Initial reset
    with download_results_lock:
        download_results_container.clear()
    with downloaded_symbols_lock:
        downloaded_symbols_registry.clear()
    
    results = {}
    
    # Pipeline queue ready for processing
    processing_queue = queue.Queue()
    
    # Queue for processing results
    processing_results = queue.Queue()
    
    # Flag to indicate when all downloads are completed
    downloads_completed = threading.Event()
    
    # Thread for managing sequential downloads
    def download_worker():
        """Worker handling all downloads in sequence"""
        for pipeline_name in pipelines_to_run:
            if pipeline_name not in config['pipelines'] or not config['pipelines'][pipeline_name].get('enabled', True):
                safe_print(f"Pipeline {pipeline_name} disabilitata, saltando", "WARNING")
                processing_results.put((pipeline_name, False))
                continue
            
            safe_print(f"Download {pipeline_name}", "DOWNLOAD")
            
            try:
                start_time = time.time()
                batches, steps = run_pipeline_download(config, pipeline_name, selected_steps)
                download_time = time.time() - start_time
                
                if batches:
                    safe_print(f"Download {pipeline_name} completed in {download_time:.1f}s", "COMPLETE")
                    # Aggiungi alla coda di processing
                    processing_queue.put((pipeline_name, batches, steps))
                else:
                    safe_print(f"No data for {pipeline_name}", "WARNING")
                    processing_results.put((pipeline_name, False))
            except Exception as e:
                safe_print(f"Download error {pipeline_name}: {e}", "ERROR")
                processing_results.put((pipeline_name, False))
        
        # Reports that all downloads are completed
        downloads_completed.set()
        safe_print("All downloads are completed", "COMPLETE")
    
    # Thread for managing processing
    def processing_worker():
        """Worker handling the processing of downloaded pipelines"""
        while True:
            try:
                # Wait for a pipeline to process or for downloads to finish
                try:
                    pipeline_name, batches, steps = processing_queue.get(timeout=5)
                except queue.Empty:
                    # If there are no items and the downloads are finished, it ends
                    if downloads_completed.is_set():
                        break
                    continue
                
                safe_print(f"Processing {pipeline_name}", "PROCESS")
                
                try:
                    start_time = time.time()
                    success = process_pipeline(config, pipeline_name, batches, steps)
                    processing_time = time.time() - start_time
                    
                    status = "SUCCESSO" if success else "FALLITO"
                    safe_print(f"Processing {pipeline_name} {status} in {processing_time:.1f}s", "COMPLETE")
                    
                    processing_results.put((pipeline_name, success))
                    
                except Exception as e:
                    safe_print(f"Processing error {pipeline_name}: {e}", "ERROR")
                    processing_results.put((pipeline_name, False))
                
                processing_queue.task_done()
                
            except Exception as e:
                safe_print(f"Processing worker error: {e}", "ERROR")
    
    # Avvia i worker
    safe_print("=== PIPELINE OVERLAP START: Parallel Downloading and Processing ===", "PHASE")
    
    # Download threads
    download_thread = threading.Thread(target=download_worker, name="DownloadWorker", daemon=True)
    download_thread.start()
    
    # Processing Thread
    processing_thread = threading.Thread(target=processing_worker, name="ProcessingWorker", daemon=True)
    processing_thread.start()
    
    # Wait until all results are ready
    results_collected = 0
    total_pipelines = len([p for p in pipelines_to_run if config['pipelines'].get(p, {}).get('enabled', True)])
    
    while results_collected < total_pipelines:
        try:
            pipeline_name, success = processing_results.get(timeout=10)
            results[pipeline_name] = success
            results_collected += 1
            safe_print(f"Collected result for {pipeline_name}: {'SUCCESS' if success else 'FAILED'}", "INFO")
        except queue.Empty:
            # Log periodico per mostrare che stiamo ancora aspettando
            #safe_print(f"In attesa risultati: {results_collected}/{total_pipelines} completati", "INFO")
            continue
    
    # Wait for thread termination
    downloads_completed.wait(timeout=10)
    processing_queue.join()
    
    download_thread.join(timeout=5)
    processing_thread.join(timeout=5)
    
    safe_print("Pipeline overlap completed", "PHASE")
    
    return results


def process_pipeline_and_signal(config, pipeline_name, batches, steps, results_dict, completion_event):
    """
    Wrapper for process_pipeline signalling completion via an event
    """
    try:
        success = process_pipeline(config, pipeline_name, batches, steps)
        with download_results_lock:  # I use the same lock to protect access to the results dictionary
            results_dict[pipeline_name] = success
        safe_print(f"Processing of {pipeline_name} successfully completed={success}", "COMPLETE")
    except Exception as e:
        safe_print(f"Error during processing of {pipeline_name}: {e}", "ERROR")
        with download_results_lock:
            results_dict[pipeline_name] = False
    finally:
        # ALWAYS report completion, even in the event of an error
        completion_event.set()

def main():
    """Main function with optimised overlapping between downloading and processing"""
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--pipelines', nargs='+', 
                       help='Pipelines to run (e.g., stocks forex futures)')
    parser.add_argument('--steps', nargs='+', 
                       choices=['download', 'outliers', 'realized_variance'],
                       help='Steps to execute (e.g., download outliers)')
    parser.add_argument('--config', default='config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', choices=['async', 'sequential'],
                       help='Execution mode (overrides config setting)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    safe_print(f"PIPELINE START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", "PHASE")
    
    try:
        # LOad configuration
        config = load_config(args.config)
        
        # Ensures that there are configuration parameters for workers
        if 'general' not in config:
            config['general'] = {}
        
        # Default values if not present
        if 'system_cores_reserved' not in config['general']:
            config['general']['system_cores_reserved'] = 1
        if 'download_threads_reserved' not in config['general']:
            config['general']['download_threads_reserved'] = 4
        
        # Determine the mode of execution
        execution_mode = args.mode or config['general'].get('execution_mode', 'sequential')
        safe_print(f"Modalità: {execution_mode}", "INFO")
        
        # Determine which pipelines to run
        if args.pipelines:
            pipelines_to_run = args.pipelines
        else:
            pipelines_to_run = [name for name, conf in config['pipelines'].items() 
                              if conf.get('enabled', True)]
        
        if not pipelines_to_run:
            safe_print("No pipeline to run. Check configuration.", "ERROR")
            sys.exit(1)
        
        safe_print(f"Pipeline to execute: {pipelines_to_run}", "INFO")
        
        # Monitor resources at the beginning
        safe_print(f"Resources: {os.cpu_count()} CPU cores, "
                  f"{psutil.virtual_memory().total / (1024**3):.1f}GB RAM", "INFO")
        
        # Run pipeline by mode
        if execution_mode.lower() == 'async':
            # Optimised execution with overlap
            results = execute_pipeline_async(config, pipelines_to_run, args.steps)
        else:
            # Sequential mode
            results = {}
            for pipeline_name in pipelines_to_run:
                if pipeline_name not in config['pipelines']:
                    safe_print(f"Pipeline '{pipeline_name}' not found. Skipping.", "WARNING")
                    continue
                
                if not config['pipelines'][pipeline_name].get('enabled', True):
                    safe_print(f"Pipeline '{pipeline_name}' disabled. Skipping.", "INFO")
                    continue
                    
                try:
                    # Run the download
                    batches, steps = run_pipeline_download(config, pipeline_name, args.steps)
                    
                    # Perform processing
                    if batches:
                        success = process_pipeline(config, pipeline_name, batches, steps)
                        results[pipeline_name] = success
                    else:
                        safe_print(f"No valid batch for {pipeline_name}", "WARNING")
                        results[pipeline_name] = False
                except Exception as e:
                    safe_print(f"Pipeline {pipeline_name} failed: {e}", "ERROR")
                    import traceback
                    traceback.print_exc()
                    results[pipeline_name] = False
        
        # Final Report
        end_time = datetime.now()
        duration = end_time - start_time
        
        safe_print("FINAL SUMMARY", "PHASE")
        safe_print(f"Mode: {execution_mode}")
        safe_print(f"Total time: {duration}")
        
        for pipeline, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            safe_print(f"{pipeline:>10}: {status}")
        
        # Determine exit code
        all_success = all(results.values()) if results else False
        if all_success:
            safe_print("\nALL PIPELINES SUCCESSFULLY COMPLETED!", "PHASE")
            sys.exit(0)
        else:
            safe_print("\nSOME PIPELINES FAILED!", "PHASE")
            sys.exit(1)
            
    except FileNotFoundError as e:
        safe_print(f"CONFIGURATION ERROR: {e}", "ERROR")
        sys.exit(1)
    except json.JSONDecodeError as e:
        safe_print(f"JSON PARSING ERROR: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        safe_print(f"CRITICAL ERRORS: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()