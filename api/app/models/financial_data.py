from pydantic import BaseModel, field_validator
from typing import Optional, Union, List, Dict
from datetime import date
from decimal import Decimal


class FinancialDataPoint(BaseModel):
    """Model for a single financial data point"""
    observation_date: date
    symbol: str
    asset_type: str
    volume: Optional[int] = None
    trades: Optional[int] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    
    # Volatility measures
    pv: Optional[float] = None
    gk: Optional[float] = None
    rr5: Optional[float] = None
    rv1: Optional[float] = None  # Questo campo potrebbe avere problemi
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
    rq1: Optional[float] = None
    rq5: Optional[float] = None
    rq5_ss: Optional[float] = None

    @field_validator('rv1', 'rv5', 'rv5_ss', 'bv1', 'bv5', 'bv5_ss', 'medrv1', 'medrv5', 'medrv5_ss', 
                    'minrv1', 'minrv5', 'minrv5_ss', 'rq1', 'rq5', 'rq5_ss', mode='before')
    @classmethod
    def convert_decimal_to_float(cls, v):
        if v is None:
            return None
        if isinstance(v, Decimal):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        return v

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
    role: str
    unlimited_access: bool = False


class DownloadResponse(BaseModel):
    """Model for download response"""
    download_url: str
    file_name: str
    expires_at: str 


class CovarianceDataPoint(BaseModel):
    """Model for a single covariance data point"""
    observation_date: date
    asset1_symbol: str
    asset2_symbol: str
    rcov: Optional[float] = None  # Realized Covariance
    rbpcov: Optional[float] = None  # Realized Bipower Covariance
    rscov_p: Optional[float] = None  # Realized Semivariance Positive
    rscov_n: Optional[float] = None  # Realized Semivariance Negative
    rscov_mp: Optional[float] = None  # Realized Semivariance Mixed Positive
    rscov_mn: Optional[float] = None  # Realized Semivariance Mixed Negative

    class Config:
        from_attributes = True


class CovarianceDataRequest(BaseModel):
    """Model for requesting covariance data with filters"""
    asset1_symbol: Optional[str] = None
    asset2_symbol: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    limit: int = 100
    fields: Optional[List[str]] = None


class CovarianceDataResponse(BaseModel):
    """Model for paginated response of covariance data"""
    data: List[CovarianceDataPoint]
    total: int
    page: int
    limit: int
    has_more: bool