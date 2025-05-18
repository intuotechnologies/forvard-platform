from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional, Set
from loguru import logger
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import uuid
from pathlib import Path

from ..core.database import get_db, query_to_dataframe
from ..core.dependencies import get_current_user, get_user_access_limits
from ..models.user import UserInDB
from ..models.financial_data import (
    FinancialDataPoint, 
    FinancialDataResponse, 
    AccessLimitResponse,
    DownloadResponse
)
from ..core.exceptions import AccessLimitExceeded, ResourceNotFound

router = APIRouter(prefix="/financial-data", tags=["financial data"])

# Create temp directory for downloads if it doesn't exist
DOWNLOAD_DIR = Path("./downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Cleanup old download files periodically (can be moved to a scheduled task)
def cleanup_old_downloads():
    """Remove download files older than 1 hour"""
    try:
        current_time = datetime.now()
        for file_path in DOWNLOAD_DIR.glob("*.csv"):
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(hours=1):
                file_path.unlink()
                logger.info(f"Removed old download file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up downloads: {e}")


def count_unique_assets(
    db: Session, 
    user_id: str, 
    asset_type: Optional[str] = None
) -> Dict[str, int]:
    """Count unique assets accessed by user by type"""
    query = """
        SELECT asset_type, COUNT(DISTINCT symbol) as count 
        FROM realized_volatility_data 
        WHERE 1=1
    """
    params = {}
    
    if asset_type:
        query += " AND asset_type = :asset_type"
        params["asset_type"] = asset_type
    
    query += " GROUP BY asset_type"
    
    result = db.execute(text(query), params).fetchall()
    return {row.asset_type: row.count for row in result}


def apply_access_limits(
    asset_type: Optional[str],
    access_limits: Dict[str, int],
    db: Session,
    user: UserInDB,
    symbols: Optional[List[str]] = None
) -> int:
    """Apply access limits based on user role and return effective limit"""
    # Admin users have no limits
    if user.role_name == "admin":
        return None  # No limit
    
    # Get counts of unique assets by type for this request
    asset_count = count_unique_assets(db, str(user.user_id), asset_type)
    
    # If requesting specific asset type, apply that limit
    if asset_type and asset_type in access_limits:
        role_limit = access_limits[asset_type]
        
        # Check if request would exceed limits
        if symbols:
            # If specific symbols requested, count if under limit
            if len(symbols) > role_limit:
                raise AccessLimitExceeded(
                    f"Request exceeds your access limit for {asset_type}: {len(symbols)} requested, limit is {role_limit}"
                )
            return len(symbols)
        
        # If not specific symbols, apply role limit
        return role_limit
    
    # If no specific asset type requested, use the most restrictive limit as default
    # This is a simplified approach - you may want different logic
    if not asset_type:
        default_limit = min(access_limits.values()) if access_limits else 100
        return default_limit
    
    # Default limit if no specific limit for this asset type
    return 100


@router.get("", response_model=FinancialDataResponse)
async def get_financial_data(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
    access_limits: Dict[str, int] = Depends(get_user_access_limits),
    symbol: Optional[str] = None,
    symbols: Optional[List[str]] = Query(None),
    asset_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    limit: int = 100,
    fields: Optional[List[str]] = Query(None)
):
    """
    Get financial data with filtering and pagination
    """
    logger.info(
        f"Financial data request by {current_user.email}",
        user_id=str(current_user.user_id),
        role=current_user.role_name,
        filters={"symbol": symbol, "symbols": symbols, "asset_type": asset_type, 
                "start_date": start_date, "end_date": end_date}
    )
    
    # Convert symbols parameter to list if needed
    symbol_list = []
    if symbol:
        symbol_list = [symbol]
    elif symbols:
        symbol_list = symbols
    
    # Apply access limits
    effective_limit = apply_access_limits(
        asset_type, 
        access_limits, 
        db, 
        current_user, 
        symbol_list
    )
    
    # Build query
    query = """
        SELECT 
            observation_date, symbol, asset_type, 
            volume, trades, open_price, close_price, 
            high_price, low_price, rv5
    """
    
    # Add requested fields if specified
    if fields:
        # Validate fields to prevent SQL injection
        valid_fields = {
            "pv", "gk", "rr5", "rv1", "rv5", "rv5_ss", "bv1", "bv5", "bv5_ss", 
            "rsp1", "rsn1", "rsp5", "rsn5", "rsp5_ss", "rsn5_ss", "medrv1", 
            "medrv5", "medrv5_ss", "minrv1", "minrv5", "minrv5_ss", "rk"
        }
        
        # Filter and validate requested fields
        valid_requested_fields = [f for f in fields if f in valid_fields]
        
        if valid_requested_fields:
            # Add fields to query
            field_str = ", ".join(valid_requested_fields)
            query = query.replace("rv5", f"rv5, {field_str}")
    
    # Complete query
    query += " FROM realized_volatility_data WHERE 1=1"
    
    # Build where clause and params
    params = {}
    
    if symbol_list:
        query += " AND symbol IN :symbols"
        params["symbols"] = tuple(symbol_list) if len(symbol_list) > 1 else tuple(symbol_list + [""])
    
    if asset_type:
        query += " AND asset_type = :asset_type"
        params["asset_type"] = asset_type
    
    if start_date:
        query += " AND observation_date >= :start_date"
        params["start_date"] = start_date
    
    if end_date:
        query += " AND observation_date <= :end_date"
        params["end_date"] = end_date
    
    # Get total count for pagination
    count_query = query.replace("SELECT \n            observation_date, symbol, asset_type, \n            volume, trades, open_price, close_price, \n            high_price, low_price, rv5", "SELECT COUNT(*)")
    
    total_count = 0
    try:
        total_count_result = db.execute(text(count_query), params).scalar()
        total_count = total_count_result or 0
    except SQLAlchemyError as e:
        logger.error(f"Error counting financial data: {e}")
        total_count = 0
    
    # Apply pagination
    offset = (page - 1) * limit
    query += " ORDER BY observation_date DESC, symbol"
    
    # Apply limit if not admin
    if effective_limit is not None:
        query += f" LIMIT {effective_limit}"
    else:
        query += " LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset
    
    try:
        # Execute query
        results = db.execute(text(query), params).fetchall()
        
        # Convert to Pydantic models
        data_points = []
        for row in results:
            data_point = {col: getattr(row, col) for col in row._mapping.keys()}
            data_points.append(FinancialDataPoint(**data_point))
        
        # Create response
        response = FinancialDataResponse(
            data=data_points,
            total=total_count,
            page=page,
            limit=limit,
            has_more=(page * limit) < total_count
        )
        
        logger.info(
            f"Returned {len(data_points)} financial data points to {current_user.email}"
        )
        
        return response
    
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving financial data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving financial data"
        )


@router.get("/limits", response_model=AccessLimitResponse)
async def get_access_limits(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
    access_limits: Dict[str, int] = Depends(get_user_access_limits)
):
    """
    Get current user's access limits and usage
    """
    if current_user.role_name == "admin":
        # Admins have unlimited access
        return AccessLimitResponse(
            limits={category: 999999 for category in access_limits},
            total_available={category: 999999 for category in access_limits},
            used={category: 0 for category in access_limits},
            remaining={category: 999999 for category in access_limits}
        )
    
    # Get count of available assets by type
    available_query = """
        SELECT asset_type, COUNT(DISTINCT symbol) as count
        FROM realized_volatility_data
        GROUP BY asset_type
    """
    
    available_results = db.execute(text(available_query)).fetchall()
    total_available = {row.asset_type: row.count for row in available_results}
    
    # Calculate used and remaining
    used = count_unique_assets(db, str(current_user.user_id))
    
    # Calculate remaining for each category
    remaining = {}
    for category, limit in access_limits.items():
        used_count = used.get(category, 0)
        remaining[category] = max(0, limit - used_count)
    
    return AccessLimitResponse(
        limits=access_limits,
        total_available=total_available,
        used=used,
        remaining=remaining
    )


@router.get("/download", response_model=DownloadResponse)
async def download_financial_data(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user),
    access_limits: Dict[str, int] = Depends(get_user_access_limits),
    symbol: Optional[str] = None,
    symbols: Optional[List[str]] = Query(None),
    asset_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[List[str]] = Query(None)
):
    """
    Generate and download financial data as CSV
    """
    logger.info(
        f"Download request by {current_user.email}",
        user_id=str(current_user.user_id),
        role=current_user.role_name,
        filters={"symbol": symbol, "symbols": symbols, "asset_type": asset_type, 
                "start_date": start_date, "end_date": end_date}
    )
    
    # Convert symbols parameter to list if needed
    symbol_list = []
    if symbol:
        symbol_list = [symbol]
    elif symbols:
        symbol_list = symbols
    
    # Apply access limits
    effective_limit = apply_access_limits(
        asset_type, 
        access_limits, 
        db, 
        current_user, 
        symbol_list
    )
    
    # Build query
    query = """
        SELECT 
            observation_date, symbol, asset_type, 
            volume, trades, open_price, close_price, 
            high_price, low_price
    """
    
    # Add all fields that exist in realized_volatility_data table
    all_fields = [
        "pv", "gk", "rr5", "rv1", "rv5", "rv5_ss", "bv1", "bv5", "bv5_ss", 
        "rsp1", "rsn1", "rsp5", "rsn5", "rsp5_ss", "rsn5_ss", "medrv1", 
        "medrv5", "medrv5_ss", "minrv1", "minrv5", "minrv5_ss", "rk"
    ]
    
    # Add all available fields to query
    field_str = ", ".join(all_fields)
    query += f", {field_str} FROM realized_volatility_data WHERE 1=1"
    
    # Build where clause and params
    params = {}
    
    if symbol_list:
        query += " AND symbol IN :symbols"
        params["symbols"] = tuple(symbol_list) if len(symbol_list) > 1 else tuple(symbol_list + [""])
    
    if asset_type:
        query += " AND asset_type = :asset_type"
        params["asset_type"] = asset_type
    
    if start_date:
        query += " AND observation_date >= :start_date"
        params["start_date"] = start_date
    
    if end_date:
        query += " AND observation_date <= :end_date"
        params["end_date"] = end_date
    
    # Order results
    query += " ORDER BY observation_date DESC, symbol"
    
    # Apply limit if not admin
    if effective_limit is not None and current_user.role_name != "admin":
        query += f" LIMIT {effective_limit}"
    
    try:
        # Get data as DataFrame
        df = query_to_dataframe(db, query, params)
        
        if df.empty:
            raise ResourceNotFound("financial data", "No data found matching the criteria")
        
        # Generate a unique filename
        file_id = uuid.uuid4()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"financial_data_{timestamp}_{file_id}.csv"
        file_path = DOWNLOAD_DIR / filename
        
        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)
        
        # Schedule cleanup of old files
        background_tasks.add_task(cleanup_old_downloads)
        
        # Calculate expiry time (1 hour)
        expiry_time = (datetime.now() + timedelta(hours=1)).isoformat()
        
        logger.info(f"Generated download file: {filename} for user {current_user.email}")
        
        # Return download info
        return DownloadResponse(
            download_url=f"/financial-data/files/{filename}",
            file_name=filename,
            expires_at=expiry_time
        )
    
    except SQLAlchemyError as e:
        logger.error(f"Database error generating download: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating data for download"
        )


@router.get("/files/{filename}")
async def get_download_file(filename: str, current_user: UserInDB = Depends(get_current_user)):
    """
    Download a previously generated file
    """
    file_path = DOWNLOAD_DIR / filename
    
    if not file_path.exists():
        logger.warning(f"Download file not found: {filename} requested by {current_user.email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Download file not found or expired"
        )
    
    # Check if file is older than 1 hour
    file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
    if file_age > timedelta(hours=1):
        file_path.unlink()  # Remove expired file
        logger.warning(f"Expired download file requested: {filename} by {current_user.email}")
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Download file has expired"
        )
    
    logger.info(f"Serving download file: {filename} to user {current_user.email}")
    
    return FileResponse(
        path=str(file_path), 
        filename=filename, 
        media_type="text/csv"
    ) 