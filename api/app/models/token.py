from pydantic import BaseModel
from typing import Optional


class Token(BaseModel):
    """Model for OAuth token response"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Model for data stored in JWT token"""
    sub: Optional[str] = None  # subject (user email)
    role: Optional[str] = None  # user role
    exp: Optional[int] = None  # expiration time 