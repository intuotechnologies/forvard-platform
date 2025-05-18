from sqladmin import Admin, ModelView
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Boolean, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from ..core.database import engine
from .auth import AdminAuth

# SQLAlchemy models for admin panels
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role_id = Column(Integer, ForeignKey("roles.role_id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to role
    role = relationship("Role", back_populates="users")


class Role(Base):
    __tablename__ = "roles"
    
    role_id = Column(Integer, primary_key=True)
    role_name = Column(String, unique=True, nullable=False)
    
    # Relationship to users
    users = relationship("User", back_populates="role")
    # Relationship to access limits
    access_limits = relationship("AssetAccessLimit", back_populates="role")


class AssetAccessLimit(Base):
    __tablename__ = "asset_access_limits"
    
    limit_id = Column(Integer, primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.role_id"))
    asset_category = Column(String, nullable=False)
    max_items = Column(Integer, nullable=False)
    
    # Relationship to role
    role = relationship("Role", back_populates="access_limits")


# Admin views
class UserAdmin(ModelView, model=User):
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-user"
    column_list = [User.user_id, User.email, User.role_id, User.created_at]
    column_details_exclude_list = [User.password_hash]
    column_searchable_list = [User.email]
    column_sortable_list = [User.email, User.created_at]
    column_formatters = {
        User.user_id: lambda m, a: str(m.user_id)[:8] + "..." if m.user_id else "",
    }
    can_export = True
    page_size = 25
    
    # Customize form options for password
    form_excluded_columns = [User.password_hash, User.created_at, User.updated_at]
    
    # Add custom field for password when creating/editing users
    form_args = {
        "role_id": {
            "label": "Role"
        }
    }


class RoleAdmin(ModelView, model=Role):
    name = "Role"
    name_plural = "Roles"
    icon = "fa-solid fa-user-tag"
    column_list = [Role.role_id, Role.role_name]
    column_searchable_list = [Role.role_name]
    can_export = True


class AssetAccessLimitAdmin(ModelView, model=AssetAccessLimit):
    name = "Access Limit"
    name_plural = "Access Limits"
    icon = "fa-solid fa-lock"
    column_list = [
        AssetAccessLimit.limit_id, 
        AssetAccessLimit.role_id, 
        AssetAccessLimit.asset_category,
        AssetAccessLimit.max_items
    ]
    can_export = True
    form_args = {
        "role_id": {
            "label": "Role"
        }
    }


# Setup function for admin
def setup_admin(app: FastAPI) -> None:
    """
    Setup admin interface
    """
    # Get session secret key from environment
    session_key = os.getenv("SESSION_SECRET_KEY", "your-super-secret-key-change-this-in-production")
    
    # Create authentication backend
    authentication_backend = AdminAuth(secret_key=session_key)
    
    # Create admin interface
    admin = Admin(
        app, 
        engine,
        authentication_backend=authentication_backend,
        title="ForVARD Admin",
        base_url="/admin",
        logo_url="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    )
    
    # Add model views
    admin.add_view(UserAdmin)
    admin.add_view(RoleAdmin)
    admin.add_view(AssetAccessLimitAdmin)
    
    return admin 