from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date


class FinancialDataPoint(BaseModel):
    """Model for a single financial data point"""
    observation_date: date
    symbol: str
    asset_type: Optional[str] = None
    volume: Optional[int] = None
    trades: Optional[int] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    pv: Optional[float] = None
    gk: Optional[float] = None
    rr5: Optional[float] = None
    rv1: Optional[float] = None
    rv5: Optional[float] = None
    rv5_ss: Optional[float] = None
    bv1: Optional[float] = None
    bv5: Optional[float] = None
    bv5_ss: Optional[float] = None
    rsp1: Optional[float] = None
    rsn1: Optional[float] = None
    rsp5: Optional[float] = None
    rsn5: Optional[float] = None
    rsp5_ss: Optional[float] = None
    rsn5_ss: Optional[float] = None
    medrv1: Optional[float] = None
    medrv5: Optional[float] = None
    medrv5_ss: Optional[float] = None
    minrv1: Optional[float] = None
    minrv5: Optional[float] = None
    minrv5_ss: Optional[float] = None
    rk: Optional[float] = None

    class Config:
        from_attributes = True


class FinancialDataRequest(BaseModel):
    """Model for requesting financial data with filters"""
    symbol: Optional[str] = None
    asset_type: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    limit: int = 100
    fields: Optional[List[str]] = None


class FinancialDataResponse(BaseModel):
    """Model for paginated response of financial data"""
    data: List[FinancialDataPoint]
    total: int
    page: int
    limit: int
    has_more: bool
    
    
class AccessLimitResponse(BaseModel):
    """Model for response containing user's access limits"""
    limits: Dict[str, int]
    total_available: Dict[str, int]
    used: Dict[str, int]
    remaining: Dict[str, int]


class DownloadResponse(BaseModel):
    """Model for download response"""
    download_url: str
    file_name: str
    expires_at: str 