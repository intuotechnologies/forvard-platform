from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import text
import uuid
from loguru import logger

from ..core.database import get_db
from ..core.security import verify_password, get_password_hash, create_access_token
from ..models.user import UserCreate, UserResponse, UserInDB
from ..models.token import Token
from ..core.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user
    """
    # Check if user already exists
    query = text("SELECT email FROM users WHERE email = :email")
    existing_user = db.execute(query, {"email": user.email}).fetchone()
    
    if existing_user:
        logger.warning(f"Registration attempt with existing email: {user.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Get role_id from role_name
    role_query = text("SELECT role_id FROM roles WHERE role_name = :role_name")
    role = db.execute(role_query, {"role_name": user.role_name}).fetchone()
    
    if not role:
        logger.error(f"Invalid role requested: {user.role_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {user.role_name}"
        )
    
    # Hash password
    hashed_password = get_password_hash(user.password)
    
    # Create new user
    user_id = uuid.uuid4()
    insert_query = text("""
        INSERT INTO users (user_id, email, password_hash, role_id)
        VALUES (:user_id, :email, :password_hash, :role_id)
        RETURNING user_id, email, created_at
    """)
    
    result = db.execute(
        insert_query, 
        {
            "user_id": user_id,
            "email": user.email,
            "password_hash": hashed_password,
            "role_id": role.role_id
        }
    ).fetchone()
    
    db.commit()
    
    logger.info(f"New user registered: {user.email} with role {user.role_name}")
    
    # Return user info
    return UserResponse(
        user_id=result.user_id,
        email=result.email,
        role_name=user.role_name,
        created_at=result.created_at
    )


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Get an access token (login)
    """
    # Look up the user
    query = text("""
        SELECT u.user_id, u.email, u.password_hash, r.role_name
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        WHERE u.email = :email
    """)
    
    user = db.execute(query, {"email": form_data.username}).fetchone()
    
    # Validate credentials
    if not user or not verify_password(form_data.password, user.password_hash):
        logger.warning(f"Failed login attempt for: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(subject=user.email, role=user.role_name)
    
    logger.info(f"User logged in: {user.email}")
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    """
    Get information about the currently logged in user
    """
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        role_name=current_user.role_name,
        created_at=current_user.created_at
    ) 