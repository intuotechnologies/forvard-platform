from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class UserBase(BaseModel):
    """Base model for user data"""
    email: EmailStr


class UserCreate(UserBase):
    """Model for user creation request"""
    password: str
    role_name: str = "base"  # Default to 'base' if not specified


class UserCredentials(UserBase):
    """Model for user login credentials"""
    password: str


class UserResponse(UserBase):
    """Model for user response data"""
    user_id: UUID
    role_name: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserInDB(UserBase):
    """Model for user data stored in the database"""
    user_id: UUID
    password_hash: str
    role_id: int
    role_name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 