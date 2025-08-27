#!/usr/bin/env python3
"""
Export Realized Covariance Data Script
ForVARD Project - Forecasting Volatility and Risk Dynamics
EU NextGenerationEU - GRINS (CUP: J43C24000210007)

This script exports the entire realized_covariance_data table from PostgreSQL,
compresses it to a ZIP file, and uploads it to MinIO S3 bucket.

Author: AI Assistant
Date: 2025-01-21
"""

import os
import sys
import logging
import psycopg2
import psycopg2.extras
import pandas as pd
import boto3
from datetime import datetime
import zipfile
from io import BytesIO
import tempfile
import wmill

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('export_covariance')


def send_slack_notification(message_text):
    """
    Sends a notification message to a configured Slack channel.
    """
        
    try:
        slack_token = wmill.get_variable("u/niccolosalvini27/SLACK_API_TOKEN")
        slack_channel = wmill.get_variable("u/niccolosalvini27/SLACK_CHANNEL_ID")

        if not slack_token or not slack_channel:
            logger.warning("Slack environment variables not found. Notification skipped.")
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

def get_db_config():
    """Get database configuration from Windmill or environment variables"""
    try:
        db_config = {
            'host': wmill.get_variable("u/niccolosalvini27/DB_HOST") or 'volare.unime.it',
            'port': int(wmill.get_variable("u/niccolosalvini27/DB_PORT") or 5432),
            'database': wmill.get_variable("u/niccolosalvini27/DB_NAME") or 'forvarddb',
            'user': wmill.get_variable("u/niccolosalvini27/DB_USER") or 'forvarduser',
            'password': wmill.get_variable("u/niccolosalvini27/DB_PASSWORD") or 'WsUpwXjEA7HHidmL8epF'
        }
        
        logger.info(f"Database config: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        return db_config
        
    except Exception as e:
        logger.error(f"Failed to get database config: {e}")
        raise

def get_s3_client():
    """Setup S3 client for MinIO"""
    try:

        s3_endpoint_url = wmill.get_variable("u/niccolosalvini27/S3_ENDPOINT_URL")
        s3_access_key = wmill.get_variable("u/niccolosalvini27/S3_ACCESS_KEY")
        s3_secret_key = wmill.get_variable("u/niccolosalvini27/S3_SECRET_KEY")
  
        
        s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name='us-east-1'
        )
        
        logger.info(f"S3 client initialized with endpoint: {s3_endpoint_url}")
        return s3_client
        
    except Exception as e:
        logger.error(f"Failed to setup S3 client: {e}")
        raise

def create_bucket_if_not_exists(s3_client, bucket_name):
    """Create S3 bucket if it doesn't exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists")
    except Exception:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket '{bucket_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create bucket '{bucket_name}': {e}")
            raise

def export_covariance_data():
    """Export realized_covariance_data table from PostgreSQL"""
    db_config = get_db_config()
    
    try:
        # Connect to database
        logger.info("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get table info first
        cursor.execute("""
            SELECT COUNT(*) as total_rows 
            FROM realized_covariance_data
        """)
        row_count = cursor.fetchone()['total_rows']
        logger.info(f"Total rows in realized_covariance_data: {row_count:,}")
        
        if row_count == 0:
            logger.warning("No data found in realized_covariance_data table")
            return None
        
        # Export data using pandas for better memory management
        logger.info("Exporting data from realized_covariance_data table...")
        query = """
            SELECT * FROM realized_covariance_data 
            ORDER BY date, asset1, asset2
        """
        
        # Use pandas to read data in chunks if needed
        df = pd.read_sql_query(query, conn)
        logger.info(f"Exported {len(df):,} rows with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Close database connection
        cursor.close()
        conn.close()
        logger.info("Database connection closed")
        
        return df
        
    except Exception as e:
        logger.error(f"Error exporting data from database: {e}")
        raise

def compress_and_upload_data(df, s3_client, bucket_name="covariance_out"):
    """Compress DataFrame to ZIP and upload to S3"""
    try:
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"realized_covariance_data_{timestamp}"
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.csv') as csv_temp:
            with tempfile.NamedTemporaryFile(suffix='.zip') as zip_temp:
                
                # Save DataFrame to CSV
                logger.info("Converting DataFrame to CSV...")
                df.to_csv(csv_temp.name, index=False)
                csv_size = os.path.getsize(csv_temp.name)
                logger.info(f"CSV file size: {csv_size / (1024*1024):.2f} MB")
                
                # Create ZIP file
                logger.info("Creating ZIP archive...")
                with zipfile.ZipFile(zip_temp.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(csv_temp.name, f"{filename}.csv")
                
                zip_size = os.path.getsize(zip_temp.name)
                compression_ratio = (1 - zip_size / csv_size) * 100
                logger.info(f"ZIP file size: {zip_size / (1024*1024):.2f} MB")
                logger.info(f"Compression ratio: {compression_ratio:.1f}%")
                
                # Upload to S3
                s3_key = f"{filename}.zip"
                logger.info(f"Uploading to S3: s3://{bucket_name}/{s3_key}")
                
                # Read ZIP file and upload
                with open(zip_temp.name, 'rb') as zip_file:
                    s3_client.put_object(
                        Bucket=bucket_name,
                        Key=s3_key,
                        Body=zip_file.read(),
                        ContentType='application/zip'
                    )
                
                logger.info(f"Successfully uploaded to S3: s3://{bucket_name}/{s3_key}")
                return s3_key, zip_size, compression_ratio
                
    except Exception as e:
        logger.error(f"Error compressing and uploading data: {e}")
        raise

def main():
    """Main function to export, compress and upload covariance data"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("COVARIANCE DATA EXPORT STARTED")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        # Setup S3 client
        s3_client = get_s3_client()
        bucket_name = "export-covariance"
        
        # Create bucket if needed
        create_bucket_if_not_exists(s3_client, bucket_name)
        
        # Export data from database
        df = export_covariance_data()
        
        if df is None or df.empty:
            logger.error("No data to export")
            return
        
        # Compress and upload
        s3_key, file_size, compression_ratio = compress_and_upload_data(df, s3_client, bucket_name)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Final summary
        logger.info("=" * 60)
        logger.info("COVARIANCE DATA EXPORT COMPLETED SUCCESSFULLY")
        logger.info(f"Records exported: {len(df):,}")
        logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.1f}%")
        logger.info(f"S3 location: s3://{bucket_name}/{s3_key}")
        logger.info(f"Execution time: {str(duration).split('.')[0]}")
        logger.info("=" * 60)
        
        # Send Slack notification
        slack_message = f"✅ *COVARIANCE DATA EXPORT COMPLETED* ✅\n\n"
        slack_message += f"*Status:* SUCCESS\n"
        slack_message += f"*Records exported:* {len(df):,}\n"
        slack_message += f"*File size:* {file_size / (1024*1024):.2f} MB\n"
        slack_message += f"*Compression:* {compression_ratio:.1f}%\n"
        slack_message += f"*S3 location:* s3://{bucket_name}/{s3_key}\n"
        slack_message += f"*Duration:* {str(duration).split('.')[0]}\n"
        slack_message += f"*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        send_slack_notification(slack_message)
        
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.error("=" * 60)
        logger.error("COVARIANCE DATA EXPORT FAILED")
        logger.error(f"Error: {e}")
        logger.error(f"Duration: {str(duration).split('.')[0]}")
        logger.error("=" * 60)
        
        # Send error notification
        slack_message = f"❌ *COVARIANCE DATA EXPORT FAILED* ❌\n\n"
        slack_message += f"*Status:* FAILURE\n"
        slack_message += f"*Error:* {str(e)}\n"
        slack_message += f"*Duration:* {str(duration).split('.')[0]}\n"
        slack_message += f"*Timestamp:* {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        send_slack_notification(slack_message)
        raise

if __name__ == "__main__":
    main()
