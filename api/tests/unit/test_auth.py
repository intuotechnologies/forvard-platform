import pytest
from fastapi import status
import json

class TestAuthEndpoints:
    """
    Tests for authentication endpoints
    """
    
    def test_login_success(self, client):
        """Test successful login with valid credentials"""
        response = client.post(
            "/auth/token",
            data={"username": "admin@example.com", "password": "adminpass"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
        assert len(response.json()["access_token"]) > 20  # Check token is not empty
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post(
            "/auth/token",
            data={"username": "admin@example.com", "password": "wrongpassword"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "detail" in response.json()
        assert response.json()["detail"] == "Incorrect email or password"
    
    def test_login_user_not_found(self, client):
        """Test login with non-existent user"""
        response = client.post(
            "/auth/token",
            data={"username": "nonexistent@example.com", "password": "password"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "detail" in response.json()
        
    def test_user_registration(self, client):
        """Test successful user registration"""
        new_user = {
            "email": "newuser@example.com",
            "password": "password123",
            "role_name": "base"
        }
        
        response = client.post(
            "/auth/register",
            json=new_user
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        assert "user_id" in response.json()
        assert response.json()["email"] == new_user["email"]
        assert response.json()["role_name"] == new_user["role_name"]
    
    def test_register_existing_email(self, client):
        """Test registration with existing email"""
        existing_user = {
            "email": "admin@example.com",
            "password": "password123",
            "role_name": "base"
        }
        
        response = client.post(
            "/auth/register",
            json=existing_user
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "detail" in response.json()
        assert "already registered" in response.json()["detail"].lower()
    
    def test_register_invalid_role(self, client):
        """Test registration with invalid role"""
        user_with_invalid_role = {
            "email": "newuser2@example.com",
            "password": "password123",
            "role_name": "invalid_role"
        }
        
        response = client.post(
            "/auth/register",
            json=user_with_invalid_role
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "detail" in response.json()
        assert "invalid role" in response.json()["detail"].lower()
    
    def test_get_current_user(self, client, test_admin_token):
        """Test getting current user info with valid token"""
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["email"] == "admin@example.com"
        assert response.json()["role_name"] == "admin"
        
    def test_get_current_user_invalid_token(self, client):
        """Test getting current user info with invalid token"""
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalidtoken"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED 