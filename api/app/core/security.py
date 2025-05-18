from datetime import datetime, timedelta
from typing import Union, Any, Optional
import os
from jose import jwt
from passlib.context import CryptContext
from pydantic import ValidationError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from ..models.token import TokenData

# Constants for security settings - better to use environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "YOUR_SUPER_SECRET_KEY_CHANGE_THIS_IN_PRODUCTION")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Setup password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 token bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def get_password_hash(password: str) -> str:
    """Generate a hashed version of the password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify if the plain password matches the hashed password"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: Union[str, Any], role: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"sub": str(subject), "role": role, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        token_data = TokenData(
            sub=payload.get("sub"),
            role=payload.get("role"),
            exp=payload.get("exp")
        )
        
        # Check if token is expired
        if token_data.exp and datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return token_data
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) 