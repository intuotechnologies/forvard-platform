"""
Compute Realized Covariance 
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Alessandra Insana
Co-author: Giulia Cruciani
Date: 01/07/2025

2025 University of Messina, Department of Economics.
Research code - Unauthorized distribution prohibited.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import logging
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time 

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load function from rcov_library
from rcov_library import process_single_day, format_covariance_output


class RCovCalculator:
    """Realized Covariance Calculator."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        self.date_pattern = re.compile(r'(\d{4})_(\d{2})_(\d{2})')
        self.exclude_pattern = re.compile(r'(_last_update|adjustment)', re.IGNORECASE)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and expand environment variables in config."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Expand environment variables
        config_str = json.dumps(config)
        for env_var in ['BASE_DIR', 'CONFIG_DIR', 'KIBOT_USER', 'KIBOT_PASSWORD']:
            config_str = config_str.replace(f"${{{env_var}}}", os.getenv(env_var, ''))
        
        return json.loads(config_str)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        if 'general' not in self.config or 'base_dir' not in self.config['general']:
            raise ValueError("Missing base_dir in configuration")

        base_dir = Path(self.config['general']['base_dir'])  
        logs_dir = base_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = logs_dir / f"rcov_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"RCov Calculator initialized: {log_file}")
        return logger
    
    def parse_symbols_file(self, symbols_file: str, asset_type: str) -> Dict[str, Dict]:
        """Parse symbols file based on asset type."""
        # Input validation - check if file exists
        if not Path(symbols_file).exists():
            self.logger.error(f"Symbols file not found: {symbols_file}")
            return {}
        
        asset_configs = {} # Main output dictionary: {symbol: {asset_type, file_format}}
        file_format = self.config['general']['file_format'] # Get global file format from config
        if asset_type == 'mixed':
            # Mixed assets: SYMBOL:TYPE or [SECTION] format
            current_section = None # Track current section for implicit typing
            
            with open(symbols_file, 'r') as f:
                for line in f:
                    line = line.strip() # Remove whitespace
                    if not line or line.startswith('#'): # Skip empty lines and comments (lines starting with #)
                        continue
                    
                    # Section headers: [STOCKS], [FOREX], etc.
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1].lower()
                        continue
                    
                    # Parse symbol
                    if ':' in line:
                        symbol, symbol_type = line.split(':', 1) # Split only on first colon
                        symbol = symbol.strip().upper() # Normalize symbol to uppercase
                        symbol_type = symbol_type.strip().lower() # Normalize type to lowercase
                    else:
                        symbol = line.upper() # Normalize symbol to uppercase
                        if current_section:
                            symbol_type = current_section
                        else:
                            # Auto-detect: simple heuristics
                            if len(symbol) == 6 and symbol.isalpha():
                                symbol_type = 'forex'
                            elif len(symbol) <= 2:
                                symbol_type = 'futures'
                            else:
                                symbol_type = 'stocks'
                    # Store symbol configuration
                    asset_configs[symbol] = {
                        'asset_type': symbol_type,
                        'file_format': file_format
                    }
            
            # Log composition
            type_counts = {}
            for config in asset_configs.values():
                atype = config['asset_type']
                type_counts[atype] = type_counts.get(atype, 0) + 1
            
            # Create summary string: "stocks:150, forex:30, futures:20"
            composition = ", ".join([f"{t}:{c}" for t, c in type_counts.items()])
            self.logger.info(f"Mixed assets: {composition}")
            
        else:
            # Handle single asset type - simple parsing
            # All symbols in file are treated as the same asset type
            with open(symbols_file, 'r') as f:
                symbols = [line.strip().upper() for line in f 
                          if line.strip() and not line.startswith('#')]
            
            for symbol in symbols:
                asset_configs[symbol] = {
                    'asset_type': asset_type,
                    'file_format': file_format
                }
            
            self.logger.info(f"Loaded {len(symbols)} {asset_type} symbols")
        
        return asset_configs
    
    '''def find_data_files(self, symbol: str, asset_type: str) -> List[str]:
        """Find data files for a symbol."""
        base_dir = Path(self.config['general']['base_dir'])
        file_format = self.config['general']['file_format']
        
        # Try different directory structures
        possible_dirs = [
            base_dir / asset_type / symbol,
            base_dir / symbol,
            base_dir / 'data' / asset_type / symbol
        ]
        
        for asset_dir in possible_dirs:
            if asset_dir.exists():
                return sorted([str(f) for f in asset_dir.glob(f"*.{file_format}")])
        
        return []
    '''
    
    def find_data_files(self, symbol: str, asset_type: str) -> List[str]:
        """Find data files for a symbol."""
        base_dir = Path(self.config['general']['base_dir'])
        file_format = self.config['general']['file_format']
        
        asset_dir = base_dir / asset_type / symbol
            
        if asset_dir.exists():

            return sorted([str(f) for f in asset_dir.glob(f"*.{file_format}")])
        
        return []
    
    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract YYYY_MM_DD date from filename using regex """
        
        # Quick check to exclude files (compiled regex: _last_update|adjustment)
        if self.exclude_pattern.search(filename):
            return None
        
        # Search for date pattern (compiled regex: (\d{4})_(\d{2})_(\d{2}))
        match = self.date_pattern.search(Path(filename).stem) # .stem delete extension
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
        date_asset_counts = {} # Dictionary to count assets per date
        total_assets = len(asset_configs) # Total number of assets to track
        
        # Count how many assets have data for each date
        for symbol, config in asset_configs.items():
            files = self.find_data_files(symbol, config['asset_type'])
            for file_path in files:
                date_str = self.extract_date_from_filename(file_path) # Extract date from filename
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
        
        # Log some statistics
        if date_asset_counts:
            all_dates_count = len([d for d, c in date_asset_counts.items() if c == total_assets])
            if all_dates_count != len(valid_dates):
                self.logger.info(f"Dates with ALL assets: {all_dates_count}")
        
        return valid_dates
    
    def get_files_for_date(self, date: str, asset_configs: Dict[str, Dict]) -> Dict[str, str]:
        """Get file paths for all assets for a specific date."""
        file_paths = {}
        
        for symbol, config in asset_configs.items():
            files = self.find_data_files(symbol, config['asset_type'])
            
            for file_path in files:
                if self.extract_date_from_filename(file_path) == date:
                    file_paths[symbol] = file_path
                    break
        
        return file_paths
    

    def check_existing_results(self, output_file: str, dates: List[str], asset_configs: Dict[str, Dict]) -> List[str]:
        """Return only dates that are completely missing."""
        if not Path(output_file).exists():
            self.logger.info("No existing results - processing all dates")
            return dates
        
        try:
            existing_df = pd.read_csv(output_file)
            if existing_df.empty:
                return dates
                
            missing_dates = []
            
            for date in dates:
                date_data = existing_df[existing_df['date'] == date]
                
                if len(date_data) == 0:
                    # missing data
                    missing_dates.append(date)
                
                    
            if missing_dates:
                complete_dates = len(dates) - len(missing_dates)
                self.logger.info(f"Found {complete_dates} existing dates, need to process {len(missing_dates)} completely new dates")
            else:
                self.logger.info("All dates exist in output file")
            
            return missing_dates
            
        except Exception as e:
            self.logger.warning(f"Error reading existing results: {e}")
            return dates
    

    def process_date_parallel(self, args):
        """Process single date (for parallel execution)."""
        date, asset_configs, pipeline_config = args
        
        # Get file paths for all assets on this specific date
        file_paths = self.get_files_for_date(date, asset_configs)
        if len(file_paths) < 2: # Need at least 2 assets for covariance calculation
            print(f"[{date}] SKIP: {len(file_paths)} assets (need ≥2)")
            return None # Skip this date if insufficient data
        
        # Extract configuration settings
        measures = pipeline_config.get('measures', ['RCov'])
        resample_freq = self.config['covariance_settings']['resample_freq']
        resample_method = self.config['covariance_settings']['resampling_method']
        early_closing_file = self.config['general'].get('early_closing_day_file')
        
        # For mixed assets, automatically apply trading hours filter
        mixed_mode = (pipeline_config['asset_type'] == 'mixed') 
        
        # Call external processing function
        cov_matrices = process_single_day(
            file_paths=file_paths,
            assets=list(asset_configs.keys()),
            date=date,
            asset_configs=asset_configs,
            resample_freq=resample_freq,
            resampling_method=resample_method,
            measures=measures,
            logger=None,  # No logger in parallel
            mixed_asset_mode=mixed_mode,
            early_closing_day_file=early_closing_file
        )
        
        # Format and return results
        if cov_matrices:
            return format_covariance_output(cov_matrices, date)
        else:
            print(f"[{date}] FAIL: processing returned empty")
            return None
    
    def find_dates_with_missing_pairs(self, output_file: str, asset_configs: Dict[str, Dict]) -> List[str]:
        """Find dates that have missing asset pairs."""
        if not Path(output_file).exists(): # Check if output file exists
            return []
        
        try:
            existing_df = pd.read_csv(output_file) # Load existing results
            # DEBUG: Controlla il contenuto
            self.logger.info(f"DEBUG find_missing_pairs: Raw first date: '{existing_df['date'].iloc[0]}'")
            self.logger.info(f"DEBUG find_missing_pairs: Unique dates count: {len(existing_df['date'].unique())}")
            if existing_df.empty:
                return []
            
            dates_with_missing_pairs = [] # List to collect problematic date
            # Calculate expected pair count
            total_assets = len(asset_configs)
            # For N assets, we need N*(N+1)/2 pairs (including diagonal)
            expected_pairs_per_date = (total_assets * (total_assets + 1)) // 2
            
            # Check each date for completeness
            for date in existing_df['date'].unique(): # Iterate through all dates in existing results
                date_data = existing_df[existing_df['date'] == date] # Filter data for current date
                
                # Check if this date has fewer pairs than expected      
                if len(date_data) < expected_pairs_per_date:
                    dates_with_missing_pairs.append(date) # Mark date as incomplete
                    # Log the discrepancy for debugging
                    self.logger.info(f"Date {date}: has {len(date_data)} pairs, needs {expected_pairs_per_date}")
            
            return sorted(dates_with_missing_pairs) # Return sorted list of problematic dates
            
        except Exception as e:
            self.logger.warning(f"Error finding missing pairs: {e}")
            return []


    def run_pipeline(self, pipeline_name: str, 
                    start_date: str = None, 
                    end_date: str = None,
                    force_recalc: bool = False) -> Optional[pd.DataFrame]:
        """Run pipeline"""
        
        # Validate pipeline
        if pipeline_name not in self.config['pipelines']:
            self.logger.error(f"Pipeline '{pipeline_name}' not found")
            return None
        
        pipeline_config = self.config['pipelines'][pipeline_name]
        
        if not pipeline_config.get('enabled', False):
            self.logger.info(f"Pipeline '{pipeline_name}' is disabled")
            return None
        
        self.logger.info(f"Running pipeline: {pipeline_name}")
        start_time = time.time()  # Start performance timer
        
        # Asset Configuration Loading
        asset_type = pipeline_config['asset_type']
        asset_configs = self.parse_symbols_file(pipeline_config['symbols_file'], asset_type)
        
        if not asset_configs:
            return None
        
        # Date Range Determination
        config_dates = pipeline_config.get('date_range', {})
        # Command line arguments override config file settings
        final_start = start_date if start_date else config_dates.get('start_date')
        final_end = end_date if end_date else config_dates.get('end_date')
        
        # Convert from MM/DD/YYYY to YYYY_MM_DD internal format
        if final_start:
            final_start = datetime.strptime(final_start, '%m/%d/%Y').strftime('%Y_%m_%d')
        if final_end:
            final_end = datetime.strptime(final_end, '%m/%d/%Y').strftime('%Y_%m_%d')
        
        # Find dates to process (only dates with sufficient asset coverage)
        # Use 1.0 for ALL assets, or 0.8 for at least 80% of assets
        dates = self.find_available_dates(asset_configs, final_start, final_end, min_assets_threshold=1.0)
        if not dates:
            self.logger.warning("No dates found")
            return None
        
        # Check existing results
        output_file = Path(self.config['general']['base_dir']) / f"rcov_{pipeline_name}.csv"
        
      
        if not force_recalc:
            dates = self.check_existing_results(str(output_file), dates, asset_configs)
            if not dates:
                # If no completely new dates, check for dates with missing pairs
                dates = self.find_dates_with_missing_pairs(str(output_file), asset_configs)
                if not dates:
                    # Everything is complete - return existing results
                    return pd.read_csv(output_file)
        
        # Process dates with incremental saving
        max_workers = min(self.config['general'].get('rv_threads_max', 4), len(dates))
        use_parallel = max_workers > 1 and len(dates) > 10
        
        self.logger.info(f"Processing {len(dates)} dates with {len(asset_configs)} assets")
        if asset_type == 'mixed':
            self.logger.info("Mixed assets: applying trading hours filter")
        if use_parallel:
            self.logger.info(f"Using {max_workers} parallel workers")
        
        # Initialize file with headers if force_recalc or file doesn't exist
        if force_recalc and output_file.exists():
            output_file.unlink()  # Delete existing file
        
        results_count = 0
        batch_size = 50  # Save every 50 processed dates
        
        if use_parallel:
            # Parallel processing with batched saving
            process_args = [(date, asset_configs, pipeline_config) for date in dates]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_date = {
                    executor.submit(self.process_date_parallel, args): args[0]
                    for args in process_args
                }
                
                batch_results = []
                completed = 0
                
                for future in as_completed(future_to_date):
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                    
                    completed += 1
                    
                    # Save batch when it reaches batch_size or at the end
                    if len(batch_results) >= batch_size or completed == len(dates):
                        if batch_results:
                            self._save_batch_results(batch_results, output_file)
                            results_count += len(batch_results)
                            batch_results = []
                    
                    if completed % 50 == 0 or completed == len(dates):
                        self.logger.info(f"Progress: {completed}/{len(dates)} processed, {results_count} saved")
        else:
            # Sequential processing with incremental saving
            batch_results = []
            
            for i, date in enumerate(dates, 1):
                file_paths = self.get_files_for_date(date, asset_configs)
                if len(file_paths) < 2: # Need at least 2 assets for covariance
                    continue

                # Check if this date already has some (but not all) pairs calculated
                if output_file.exists():
                    try:
                        existing_df = pd.read_csv(output_file)
                        date_data = existing_df[existing_df['date'] == date]
                        
                        if len(date_data) > 0:
                            # Calculate which pairs are missing for this date
                            existing_pairs = set((row['asset1'], row['asset2']) for _, row in date_data.iterrows())
                            expected_pairs = set()
                            available_assets = list(asset_configs.keys())
                            
                            # Generate all expected pairs (upper triangle + diagonal)
                            for idx1, asset1 in enumerate(available_assets):
                                for idx2, asset2 in enumerate(available_assets):
                                    if idx1 <= idx2:
                                        expected_pairs.add((asset1, asset2))
                            
                            missing_pairs = expected_pairs - existing_pairs
                            
                            if not missing_pairs:
                                # Date is complete - skip
                                continue
                            else:
                                self.logger.info(f"Date {date}: adding {len(missing_pairs)} missing pairs")
                    except:
                        pass  # If error reading file, process anyway
                
                # Extract processing configuration
                measures = pipeline_config.get('measures', ['RCov'])
                resample_freq = self.config['covariance_settings']['resample_freq']
                resample_method = self.config['covariance_settings']['resampling_method']
                early_closing_file = self.config['general'].get('early_closing_day_file')
                mixed_mode = (asset_type == 'mixed')
                
                cov_matrices = process_single_day(
                    file_paths=file_paths,
                    assets=list(asset_configs.keys()),
                    date=date,
                    asset_configs=asset_configs,
                    resample_freq=resample_freq,
                    resampling_method=resample_method,
                    measures=measures,
                    logger=self.logger,
                    mixed_asset_mode=mixed_mode,
                    early_closing_day_file=early_closing_file
                )
                
                # Handle partial results for incremental updates
                if cov_matrices:
                    
                    if output_file.exists():
                        try:
                            existing_df = pd.read_csv(output_file)
                            date_data = existing_df[existing_df['date'] == date]
                            
                            if len(date_data) > 0:
                                # Date has partial data - only add missing pairs
                                existing_pairs = set((row['asset1'], row['asset2']) for _, row in date_data.iterrows())
                                all_formatted = format_covariance_output(cov_matrices, date)
                                
                                # Filter to only new pairs
                                missing_rows = []
                                for _, row in all_formatted.iterrows():
                                    pair = (row['asset1'], row['asset2'])
                                    if pair not in existing_pairs:
                                        missing_rows.append(row.to_dict())
                                
                                if missing_rows:
                                    formatted_result = pd.DataFrame(missing_rows)
                                    batch_results.append(formatted_result)
                                    self.logger.info(f"Adding {len(missing_rows)} missing pairs for {date}")
                            else:
                                # Completely new date
                                batch_results.append(format_covariance_output(cov_matrices, date))
                        except:
                            # Error reading - treat as new date
                            batch_results.append(format_covariance_output(cov_matrices, date))
                    else:
                        # File doesn't exist - treat as new date 
                        batch_results.append(format_covariance_output(cov_matrices, date))
                    
                
                # Save batch when it reaches batch_size or at the end
                if len(batch_results) >= batch_size or i == len(dates):
                    if batch_results:
                        self._save_batch_results(batch_results, output_file)
                        results_count += len(batch_results)
                        batch_results = []
                
                if i % 10 == 0 or i == len(dates):
                    self.logger.info(f"Progress: {i}/{len(dates)} processed, {results_count} saved")
        
        # Final Result Processing and Sorting
        if output_file.exists() and results_count > 0:
            df = pd.read_csv(output_file)
            
            # Complex sorting: by date first, then by asset pair order
            df_sorted = []
            original_asset_order = list(asset_configs.keys()) #nnnn
            # Sort by date chronologically
            for date in sorted(df['date'].unique(), key=lambda x: pd.to_datetime(x, format='%Y_%m_%d')):
                date_df = df[df['date'] == date]
                
                # Custom sorting function for asset pairs
                def sort_key(row):
                    try:
                        idx1 = original_asset_order.index(row['asset1'])
                        idx2 = original_asset_order.index(row['asset2'])
                        return (idx1, idx2) # Sort by asset order
                    except ValueError:
                        return (999, 999)  # Unknown assets go to end
                
                # Sort pairs within each date
                date_df = date_df.iloc[date_df.apply(sort_key, axis=1).argsort()].reset_index(drop=True)
                df_sorted.append(date_df)
        
                
            # Combine all sorted date chunks
            df = pd.concat(df_sorted, ignore_index=True)
            df.to_csv(output_file, index=False)
           
            
            # Success metrics
            success_rate = results_count / len(dates) * 100
            self.logger.info(f"SUCCESS: {len(df)} total records in {output_file.name}")
            self.logger.info(f"Success rate: {success_rate:.1f}%")
            self.logger.info(f"Total execution time: {time.time() - start_time:.1f} seconds")  
            return df
        else:
            self.logger.warning("No successful results")
            return None
    
    def _save_batch_results(self, batch_results: List[pd.DataFrame], output_file: Path):
        """Save a batch of results to CSV file."""
        if not batch_results: # Guard against empty batches
            return
        
        # Combine all DataFrames in the batch into a single DataFrame
        batch_df = pd.concat(batch_results, ignore_index=True)
        
        # Append to existing file or create new one
        if output_file.exists():
            batch_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(output_file, mode='w', header=True, index=False)

 
  
def main():
    parser = argparse.ArgumentParser(description='Realized Covariance Calculator')
    parser.add_argument('--config', required=True, help='Configuration JSON file')
    parser.add_argument('--pipeline', help='Pipeline to run')
    parser.add_argument('--start-date', help='Start date (MM/DD/YYYY)')
    parser.add_argument('--end-date', help='End date (MM/DD/YYYY)')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    parser.add_argument('--list-pipelines', action='store_true', help='List pipelines')
    
    args = parser.parse_args()
    
    try:
        calculator = RCovCalculator(args.config)
        
        if args.list_pipelines:
            print("Available pipelines:")
            for name, config in calculator.config['pipelines'].items():
                status = "enabled" if config.get('enabled', False) else "disabled"
                asset_type = config.get('asset_type', 'unknown')
                measures = config.get('measures', [])
                date_range = config.get('date_range')
                
                print(f"{name} ({asset_type}) - {status}")
                print(f"Measures: {measures}")
                if date_range:
                    print(f"Dates: {date_range.get('start_date')} to {date_range.get('end_date')}")
                else:
                    print(f"Dates: all available")
                print()
            return 0
        
        if args.pipeline:
            result = calculator.run_pipeline(
                args.pipeline, 
                args.start_date, 
                args.end_date,
                args.force
            )
            
            if result is not None:
                print(f"SUCCESS: {len(result)} covariance records processed")
                return 0
            else:
                print("FAILED")
                return 1
        else:
            # Run all enabled pipelines (like original behavior)
            enabled_pipelines = {
                name: config for name, config in calculator.config['pipelines'].items()
                if config.get('enabled', False)
            }
            
            if not enabled_pipelines:
                print("No enabled pipelines found")
                return 1
            
            print(f"Running {len(enabled_pipelines)} enabled pipelines:")
            all_success = True
            
            for pipeline_name in enabled_pipelines:
                print(f"\n--- Processing pipeline: {pipeline_name} ---")
                
                result = calculator.run_pipeline(
                    pipeline_name,
                    args.start_date,
                    args.end_date,
                    args.force
                )
                
                if result is not None:
                    print(f"{pipeline_name}: {len(result)} records")
                else:
                    print(f"{pipeline_name}: FAILED")
                    all_success = False
            
            print(f"\n{'SUCCESS' if all_success else 'PARTIAL SUCCESS'}: All enabled pipelines processed")
            return 0 if all_success else 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())