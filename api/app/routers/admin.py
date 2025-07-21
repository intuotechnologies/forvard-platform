from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
from loguru import logger

from ..core.database import get_db
from ..core.dependencies import get_current_user, verify_admin_role
from ..models.user import UserInDB, UserResponse

router = APIRouter(prefix="/api/admin", tags=["admin"], responses={
    status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
    status.HTTP_403_FORBIDDEN: {"description": "Forbidden - Admin access required"},
})


@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    db: Session = Depends(get_db),
    admin_user: UserInDB = Depends(verify_admin_role)
):
    """
    Get all users (admin only)
    """
    logger.info(f"Admin {admin_user.email} requested user list")
    
    query = text("""
        SELECT u.user_id, u.email, r.role_name, u.created_at
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        ORDER BY u.created_at DESC
    """)
    
    results = db.execute(query).fetchall()
    
    users = []
    for row in results:
        users.append(UserResponse(
            user_id=row.user_id,
            email=row.email,
            role_name=row.role_name,
            created_at=row.created_at
        ))
    
    return users


@router.get("/roles")
async def get_all_roles(
    db: Session = Depends(get_db),
    admin_user: UserInDB = Depends(verify_admin_role)
):
    """
    Get all roles (admin only)
    """
    logger.info(f"Admin {admin_user.email} requested roles list")
    
    query = text("SELECT role_id, role_name FROM roles ORDER BY role_id")
    results = db.execute(query).fetchall()
    
    roles = []
    for row in results:
        roles.append({
            "role_id": row.role_id,
            "role_name": row.role_name
        })
    
    return roles


@router.get("/stats")
async def get_admin_stats(
    db: Session = Depends(get_db),
    admin_user: UserInDB = Depends(verify_admin_role)
):
    """
    Get admin statistics (admin only)
    """
    logger.info(f"Admin {admin_user.email} requested admin stats")
    
    # Get user count by role
    user_stats_query = text("""
        SELECT r.role_name, COUNT(u.user_id) as count
        FROM roles r
        LEFT JOIN users u ON r.role_id = u.role_id
        GROUP BY r.role_id, r.role_name
        ORDER BY r.role_id
    """)
    
    user_stats = db.execute(user_stats_query).fetchall()
    
    # Get asset count by category
    asset_stats_query = text("""
        SELECT asset_category, COUNT(*) as count
        FROM assets
        GROUP BY asset_category
        ORDER BY asset_category
    """)
    
    try:
        asset_stats = db.execute(asset_stats_query).fetchall()
    except Exception:
        # If assets table doesn't exist
        asset_stats = []
    
    # Get financial data count
    data_stats_query = text("""
        SELECT asset_type, COUNT(*) as count
        FROM realized_volatility_data
        GROUP BY asset_type
        ORDER BY asset_type
    """)
    
    data_stats = db.execute(data_stats_query).fetchall()
    
    return {
        "users_by_role": [{"role": row.role_name, "count": row.count} for row in user_stats],
        "assets_by_category": [{"category": row.asset_category, "count": row.count} for row in asset_stats],
        "financial_data_by_type": [{"type": row.asset_type, "count": row.count} for row in data_stats]
    }


@router.get("/permissions/{user_id}")
async def get_user_permissions(
    user_id: str,
    db: Session = Depends(get_db),
    admin_user: UserInDB = Depends(verify_admin_role)
):
    """
    Get permissions for a specific user (admin only)
    """
    logger.info(f"Admin {admin_user.email} requested permissions for user {user_id}")
    
    # Get user info
    user_query = text("""
        SELECT u.user_id, u.email, r.role_name
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        WHERE u.user_id = :user_id
    """)
    
    user_result = db.execute(user_query, {"user_id": user_id}).fetchone()
    if not user_result:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's asset permissions
    permissions_query = text("""
        SELECT a.asset_category, COUNT(*) as count, 
               array_agg(a.symbol ORDER BY a.symbol) as symbols
        FROM asset_permissions ap
        JOIN assets a ON ap.asset_id = a.asset_id
        WHERE ap.user_id = :user_id
        GROUP BY a.asset_category
        ORDER BY a.asset_category
    """)
    
    try:
        permissions = db.execute(permissions_query, {"user_id": user_id}).fetchall()
    except Exception:
        permissions = []
    
    return {
        "user": {
            "user_id": str(user_result.user_id),
            "email": user_result.email,
            "role": user_result.role_name
        },
        "permissions": [
            {
                "category": row.asset_category,
                "count": row.count,
                "symbols": row.symbols[:10] if row.symbols else []  # Show first 10 symbols
            } for row in permissions
        ]
    } 