from sqladmin.authentication import AuthenticationBackend
from fastapi import Request, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from jose import jwt
from datetime import datetime, timedelta
import os

from ..core.database import get_db
from ..core.security import oauth2_scheme, decode_token, JWT_SECRET_KEY, JWT_ALGORITHM

class AdminAuth(AuthenticationBackend):
    """Authentication backend for admin panel"""
    
    async def login(self, request: Request) -> bool:
        """
        Handle admin login
        """
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        
        if not username or not password:
            return False
        
        # Get DB session
        db = next(get_db())
        
        try:
            # Find user with admin role
            query = text("""
                SELECT u.user_id, u.email, u.password_hash, r.role_name
                FROM users u
                JOIN roles r ON u.role_id = r.role_id
                WHERE u.email = :email AND r.role_name = 'admin'
            """)
            
            result = db.execute(query, {"email": username}).fetchone()
            
            if not result:
                return False
            
            # Verify password (in a real production app, use a secure password check here)
            # This is a placeholder implementation
            from ..core.security import verify_password
            if not verify_password(password, result.password_hash):
                return False
            
            # Create admin token
            expiry = datetime.utcnow() + timedelta(hours=1)
            token = jwt.encode(
                {
                    "sub": username,
                    "role": "admin",
                    "exp": expiry
                },
                JWT_SECRET_KEY,
                algorithm=JWT_ALGORITHM
            )
            
            # Set token in session
            request.session.update({"admin_token": token})
            return True
            
        except Exception as e:
            print(f"Admin login error: {str(e)}")
            return False
        finally:
            db.close()

    async def logout(self, request: Request) -> bool:
        """
        Handle admin logout
        """
        request.session.pop("admin_token", None)
        return True

    async def authenticate(self, request: Request) -> bool:
        """
        Verify user is authenticated and has admin role
        """
        token = request.session.get("admin_token")
        
        if not token:
            return False
            
        try:
            # Decode and validate token
            payload = jwt.decode(
                token, 
                JWT_SECRET_KEY, 
                algorithms=[JWT_ALGORITHM]
            )
            
            # Check if token has expired
            expiry = payload.get("exp")
            if not expiry or datetime.fromtimestamp(expiry) < datetime.utcnow():
                return False
                
            # Verify the user has admin role
            if payload.get("role") != "admin":
                return False
                
            return True
            
        except jwt.JWTError:
            return False 