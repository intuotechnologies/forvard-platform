from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from loguru import logger
from typing import Dict, Optional

from .database import get_db
from .security import oauth2_scheme, decode_token
from ..models.user import UserInDB


async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> UserInDB:
    """
    Get the current authenticated user from JWT token
    """
    # Decode the token
    token_data = decode_token(token)
    if not token_data.sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    email = token_data.sub
    query = text("""
        SELECT u.user_id, u.email, u.password_hash, u.role_id, r.role_name, 
               u.created_at, u.updated_at
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        WHERE u.email = :email
    """)
    
    result = db.execute(query, {"email": email}).fetchone()
    if not result:
        logger.warning(f"User with email {email} not found but had valid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if the role matches
    if token_data.role and token_data.role != result.role_name:
        logger.warning(f"Role mismatch for user {email}: token={token_data.role}, db={result.role_name}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Role mismatch",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create user object
    user = {
        "user_id": result.user_id,
        "email": result.email,
        "password_hash": result.password_hash,
        "role_id": result.role_id,
        "role_name": result.role_name,
        "created_at": result.created_at,
        "updated_at": result.updated_at
    }
    
    return UserInDB(**user)


async def get_user_access_limits(
    db: Session = Depends(get_db),
    current_user: UserInDB = Depends(get_current_user)
) -> Dict[str, int]:
    """
    Get the access limits for the current user based on their role
    """
    query = text("""
        SELECT asset_category, max_items 
        FROM asset_access_limits
        WHERE role_id = :role_id
    """)
    
    results = db.execute(query, {"role_id": current_user.role_id}).fetchall()
    return {row.asset_category: row.max_items for row in results}


async def verify_admin_role(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """
    Verify that the current user has admin role
    """
    if current_user.role_name != "admin":
        logger.warning(f"User {current_user.email} attempted to access admin endpoint without admin role")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current_user 