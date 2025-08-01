"""
ForVARD Project - Database Backup Script
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

Author: Database Backup Script
Date: 2025

This script creates database dumps for both development and production environments
and saves them to the /backup directory with proper naming and organization.
"""

import os
import subprocess
import logging
from datetime import datetime
import wmill
import requests


def setup_logging():
    """Setup logging for the backup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def send_slack_notification(message_text):
    """
    Sends a notification message to a configured Slack channel.
    """
    try:
        slack_token = wmill.get_variable("u/niccolosalvini27/SLACK_API_TOKEN")
        slack_channel = wmill.get_variable("u/niccolosalvini27/SLACK_CHANNEL_ID")

        if not slack_token or not slack_channel:
            logger.warning("Slack environment variables not found. Skipping notification.")
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
            logger.info("Slack notification sent successfully.")
        else:
            logger.error(f"Slack API error: {response_json.get('error')}")

    except Exception as e:
        logger.warning(f"Error sending Slack notification: {e}")


def create_backup_directory(base_path="/backup"):
    """Create backup directory structure"""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Create main backup directory
    backup_dir = os.path.join(base_path, "database_backups", timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create subdirectories for dev and prod
    dev_dir = os.path.join(backup_dir, "dev")
    prod_dir = os.path.join(backup_dir, "prod")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(prod_dir, exist_ok=True)
    
    return backup_dir, dev_dir, prod_dir, timestamp


def backup_database(host, port, user, password, database, output_file, logger):
    """
    Create a database backup using pg_dump
    
    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        database: Database name
        output_file: Output file path
        logger: Logger instance
    """
    try:
        # Set environment variable for password
        env = os.environ.copy()
        env['PGPASSWORD'] = password
        
        # Construct pg_dump command
        cmd = [
            'pg_dump',
            f'--host={host}',
            f'--port={port}',
            f'--username={user}',
            '--verbose',
            '--clean',
            '--no-owner',
            '--no-privileges',
            '--format=custom',
            f'--file={output_file}',
            database
        ]
        
        logger.info(f"Starting backup for database: {database}")
        logger.info(f"Output file: {output_file}")
        
        # Execute pg_dump
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            # Get file size
            file_size = os.path.getsize(output_file)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"Backup completed successfully for {database}")
            logger.info(f"File size: {file_size_mb:.2f} MB")
            return True, file_size_mb
        else:
            logger.error(f"Backup failed for {database}")
            logger.error(f"Error: {result.stderr}")
            return False, 0
            
    except subprocess.TimeoutExpired:
        logger.error(f"Backup timeout for {database}")
        return False, 0
    except Exception as e:
        logger.error(f"Unexpected error during backup of {database}: {e}")
        return False, 0


def cleanup_old_backups(base_path="/backup", keep_days=7):
    """
    Clean up old backup directories, keeping only the last N days
    
    Args:
        base_path: Base backup path
        keep_days: Number of days to keep backups
    """
    try:
        backup_root = os.path.join(base_path, "database_backups")
        if not os.path.exists(backup_root):
            return
        
        # Get all backup directories
        backup_dirs = []
        for item in os.listdir(backup_root):
            item_path = os.path.join(backup_root, item)
            if os.path.isdir(item_path):
                backup_dirs.append((item, item_path))
        
        # Sort by name (timestamp)
        backup_dirs.sort(reverse=True)
        
        # Keep only the most recent ones
        if len(backup_dirs) > keep_days:
            for _, dir_path in backup_dirs[keep_days:]:
                logger.info(f"Removing old backup: {dir_path}")
                subprocess.run(['rm', '-rf', dir_path])
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


def main(
    host="volare.unime.it",
    port=5432,
    user="forvarduser",
    password="WsUpwXjEA7HHidmL8epF",
    dev_database="forvarddb_dev",
    prod_database="forvarddb",
    backup_base_path="/backup",
    keep_backups_days=7,
    enable_cleanup=True
):
    """
    Main function to backup both development and production databases
    
    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        dev_database: Development database name
        prod_database: Production database name
        backup_base_path: Base path for backups
        keep_backups_days: Number of days to keep old backups
        enable_cleanup: Whether to enable cleanup of old backups
    """
    global logger
    logger = setup_logging()
    
    start_time = datetime.now()
    logger.info(f"DATABASE BACKUP START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create backup directory structure
        backup_dir, dev_dir, prod_dir, timestamp = create_backup_directory(backup_base_path)
        logger.info(f"Backup directory created: {backup_dir}")
        
        # Backup results
        backup_results = {}
        
        # Backup development database
        dev_output_file = os.path.join(dev_dir, f"{dev_database}_{timestamp}.dump")
        dev_success, dev_size = backup_database(
            host, port, user, password, dev_database, dev_output_file, logger
        )
        backup_results['dev'] = {
            'success': dev_success,
            'size_mb': dev_size,
            'file': dev_output_file
        }
        
        # Backup production database
        prod_output_file = os.path.join(prod_dir, f"{prod_database}_{timestamp}.dump")
        prod_success, prod_size = backup_database(
            host, port, user, password, prod_database, prod_output_file, logger
        )
        backup_results['prod'] = {
            'success': prod_success,
            'size_mb': prod_size,
            'file': prod_output_file
        }
        
        # Cleanup old backups if enabled
        if enable_cleanup:
            cleanup_old_backups(backup_base_path, keep_backups_days)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        successful_backups = sum(1 for result in backup_results.values() if result['success'])
        total_size = sum(result['size_mb'] for result in backup_results.values())
        
        logger.info("BACKUP SUMMARY")
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Successful backups: {successful_backups}/{len(backup_results)}")
        logger.info(f"Total backup size: {total_size:.2f} MB")
        
        # Determine status
        if all(result['success'] for result in backup_results.values()):
            status_emoji = "✅"
            status_text = "SUCCESS"
            logger.info("ALL DATABASE BACKUPS COMPLETED SUCCESSFULLY!")
        else:
            status_emoji = "⚠️" if successful_backups > 0 else "❌"
            status_text = "PARTIAL FAILURE" if successful_backups > 0 else "FAILURE"
            logger.warning("SOME DATABASE BACKUPS FAILED!")
        
        # Create Slack message
        slack_message = f"{status_emoji} **DATABASE BACKUP COMPLETED** {status_emoji}\n\n"
        slack_message += f"**Status:** {status_text}\n"
        slack_message += f"**Duration:** {str(duration).split('.')[0]}\n"
        slack_message += f"**Backup Location:** {backup_dir}\n\n"
        
        slack_message += "**Backup Results:**"
        for env, result in backup_results.items():
            result_emoji = "✅" if result['success'] else "❌"
            status = "SUCCESS" if result['success'] else "FAILED"
            size_info = f" ({result['size_mb']:.2f} MB)" if result['success'] else ""
            slack_message += f"\n{result_emoji} {env.upper()}: {status}{size_info}"
        
        slack_message += f"\n\n**Total Size:** {total_size:.2f} MB"
        slack_message += f"\n**Successful:** {successful_backups}/{len(backup_results)}"
        slack_message += f"\n\n**Timestamp:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send Slack notification
        send_slack_notification(slack_message)
        
        return {
            'success': all(result['success'] for result in backup_results.values()),
            'results': backup_results,
            'backup_dir': backup_dir,
            'total_size_mb': total_size,
            'duration': str(duration)
        }
        
    except Exception as e:
        error_msg = f"CRITICAL ERROR DURING BACKUP: {e}"
        logger.error(error_msg)
        
        # Send error notification
        slack_message = f"❌ **DATABASE BACKUP FAILED** ❌\n\n"
        slack_message += f"**Error:** {str(e)}\n"
        slack_message += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_slack_notification(slack_message)
        
        raise


if __name__ == "__main__":
    # Example usage
    result = main()
    print(f"Backup completed: {result}") 