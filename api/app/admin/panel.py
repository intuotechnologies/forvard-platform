from sqladmin import Admin, ModelView, BaseView
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, Boolean, Table, func, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from ..core.database import engine
from .auth import AdminAuth
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired
import uuid
import logging
from fastapi import Request
from fastapi.responses import RedirectResponse
from pathlib import Path

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
    
    def __str__(self):
        return self.email


class Role(Base):
    __tablename__ = "roles"
    
    role_id = Column(Integer, primary_key=True)
    role_name = Column(String, unique=True, nullable=False)
    
    # Relationship to users
    users = relationship("User", back_populates="role")
    # Relationship to access limits
    access_limits = relationship("AssetAccessLimit", back_populates="role")
    
    def __str__(self):
        return self.role_name


class AssetAccessLimit(Base):
    __tablename__ = "asset_access_limits"
    
    limit_id = Column(Integer, primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.role_id"))
    asset_category = Column(String, nullable=False)
    max_items = Column(Integer, nullable=False)
    
    # Relationship to role
    role = relationship("Role", back_populates="access_limits")
    
    def __str__(self):
        return f"{self.asset_category} ({self.max_items})"


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
    
    # Customize form options
    form_excluded_columns = [User.password_hash, User.created_at, User.updated_at]
    
    # Add form args
    form_args = {
        "role_id": {
            "label": "Role"
        }
    }
    
    column_labels = {
        User.role_id: "Role"
    }
    
    # Show relationship using role name
    column_formatters_detail = {
        User.role_id: lambda m, a: m.role.role_name if m.role else ""
    }
    
    # Override form creation to add password field
    async def scaffold_form(self):
        form_class = await super().scaffold_form()
        form_class.password = PasswordField(
            "Password", 
            validators=[DataRequired()],
            render_kw={
                "class": "form-control", 
                "placeholder": "Enter a password for this user",
                "autocomplete": "new-password",
                "style": "border: 2px solid #007bff;"
            }
        )
        return form_class
    
    async def insert_model(self, request, data):
        """Override insert_model to handle password hashing"""
        # Log di debug
        logger = logging.getLogger(__name__)
        logger.info(f"Insert model called with data keys: {list(data.keys())}")
        
        # Handle password
        if "password" in data and data["password"]:
            from ..core.security import get_password_hash
            data["password_hash"] = get_password_hash(data["password"])
            logger.info(f"Password hash creato dalla password inserita")
            del data["password"]
        else:
            # Se non c'è password, imposta un valore di default
            logger.warning("Password non trovata nei dati!")
            from ..core.security import get_password_hash
            data["password_hash"] = get_password_hash("default_password")
            logger.info(f"Password hash di default impostato")
        
        # Generate UUID if not provided
        if "user_id" not in data or not data["user_id"]:
            data["user_id"] = str(uuid.uuid4())
            
        return await super().insert_model(request, data)
        
    async def after_model_change(self, data, model, is_created):
        """Log dopo la creazione del modello"""
        logger = logging.getLogger(__name__)
        if is_created:
            logger.info(f"Nuovo utente creato: {model.email}")
            if hasattr(model, 'password') and not model.password:
                logger.info("Utente creato con password predefinita. Chiedi all'amministratore per il reset.")


class RoleAdmin(ModelView, model=Role):
    name = "Role"
    name_plural = "Roles"
    icon = "fa-solid fa-user-tag"
    column_list = [Role.role_id, Role.role_name]
    column_searchable_list = [Role.role_name]
    can_export = True
    
    # Better display for relationships
    column_formatters_detail = {
        "users": lambda m, a: ", ".join([user.email for user in m.users]) if m.users else "",
        "access_limits": lambda m, a: ", ".join([f"{limit.asset_category} ({limit.max_items})" for limit in m.access_limits]) if m.access_limits else ""
    }


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
    
    # Add form args to better handle role selection
    form_args = {
        "role_id": {
            "label": "Role"
        },
        "asset_category": {
            "label": "Asset Category"
        },
        "max_items": {
            "label": "Maximum Items"
        }
    }
    
    # Format the role_id column more safely - avoid detached instance errors
    column_formatters = {
        AssetAccessLimit.role_id: lambda m, a: str(m.role_id)
    }
    
    column_labels = {
        AssetAccessLimit.role_id: "Role",
        AssetAccessLimit.asset_category: "Asset Category",
        AssetAccessLimit.max_items: "Maximum Items"
    }

    # Use eager loading to avoid DetachedInstanceError
    column_select_related_list = ["role"]
    
    # Add safe error handling for model operations
    async def insert_model(self, request, data):
        """Handle insert with better error handling"""
        logger = logging.getLogger(__name__)
        logger.info(f"Inserting asset access limit with data: {data}")
        try:
            return await super().insert_model(request, data)
        except Exception as e:
            logger.error(f"Error inserting asset access limit: {str(e)}")
            raise
    
    async def update_model(self, request, model, data):
        """Handle update with better error handling"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating asset access limit ID {model.limit_id} with data: {data}")
        try:
            return await super().update_model(request, model, data)
        except Exception as e:
            logger.error(f"Error updating asset access limit: {str(e)}")
            raise
    
    async def delete_model(self, request, model):
        """Handle delete with better error handling"""
        logger = logging.getLogger(__name__)
        logger.info(f"Deleting asset access limit ID {model.limit_id}")
        try:
            return await super().delete_model(request, model)
        except Exception as e:
            logger.error(f"Error deleting asset access limit: {str(e)}")
            raise
            
    # Override scaffold_form to customize the form
    async def scaffold_form(self):
        form_class = await super().scaffold_form()
        
        # Add custom validation for asset_category - fix the check for field existence
        if hasattr(form_class, 'asset_category'):
            form_class.asset_category.validators = [DataRequired()]
            form_class.asset_category.render_kw = {"placeholder": "Enter asset category (e.g., stocks, futures, exchange_rates)"}
            
        # Add custom validation for max_items
        if hasattr(form_class, 'max_items'):
            form_class.max_items.validators = [DataRequired()]
            form_class.max_items.render_kw = {"placeholder": "Enter maximum allowed items"}
            
        return form_class


# Setup function for admin
def setup_admin(app: FastAPI) -> None:
    """
    Setup admin interface
    """
    # Get session secret key from environment
    session_key = os.getenv("SESSION_SECRET_KEY", "your-super-secret-key-change-this-in-production")
    
    # Create authentication backend
    authentication_backend = AdminAuth(secret_key=session_key)
    
    # Setup logging for admin
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create admin interface with enhanced configuration
    admin = Admin(
        app, 
        engine,
        authentication_backend=authentication_backend,
        title="ForVARD Admin",
        base_url="/admin",
        logo_url="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
        debug=True  # Enable debug mode for better error reporting
    )
    
    # Add model views
    admin.add_view(UserAdmin)
    admin.add_view(RoleAdmin)
    admin.add_view(AssetAccessLimitAdmin)
    
    logger.info("Admin panel successfully configured")
    
    return admin