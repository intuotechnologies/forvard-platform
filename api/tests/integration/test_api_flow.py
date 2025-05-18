import pytest
import uuid
from fastapi import status

class TestAPIUserFlow:
    """
    Integration tests for typical API usage flows
    """
    
    def test_registration_and_login_flow(self, client):
        """Test the registration and login flow for a new user"""
        # Step 1: Register a new user with a unique email
        unique_id = str(uuid.uuid4())[:8]
        new_user = {
            "email": f"test_{unique_id}@example.com",
            "password": "securepassword123",
            "role_name": "base"
        }
        
        register_response = client.post(
            "/auth/register",
            json=new_user
        )
        
        assert register_response.status_code == status.HTTP_201_CREATED
        assert register_response.json()["email"] == new_user["email"]
        
        # Step 2: Login with new user
        login_response = client.post(
            "/auth/token",
            data={"username": new_user["email"], "password": new_user["password"]}
        )
        
        assert login_response.status_code == status.HTTP_200_OK
        assert "access_token" in login_response.json()
        
        token = login_response.json()["access_token"]
        
        # Step 3: Get current user info
        me_response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert me_response.status_code == status.HTTP_200_OK
        assert me_response.json()["email"] == new_user["email"]
        assert me_response.json()["role_name"] == new_user["role_name"]
        
        return token
    
    def test_data_access_flow(self, client):
        """Test the complete data access flow for a user"""
        # Start with registration
        token = self.test_registration_and_login_flow(client)
        
        # Step 1: Check access limits
        limits_response = client.get(
            "/financial-data/limits", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert limits_response.status_code == status.HTTP_200_OK
        assert limits_response.json()["role"] == "base"
        assert "stock" in limits_response.json()["limits"]
        
        # Step 2: Get financial data with limits
        data_response = client.get(
            "/financial-data?asset_type=stock&symbol=AAPL", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert data_response.status_code == status.HTTP_200_OK
        assert "data" in data_response.json()
        assert len(data_response.json()["data"]) > 0
        
        # Step 3: Download allowed data
        download_response = client.get(
            "/financial-data/download?asset_type=stock&symbol=AAPL", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert download_response.status_code == status.HTTP_200_OK
        assert "download_url" in download_response.json()
        
        # Step 4: Access the download file
        file_url = download_response.json()["download_url"].split("/")[-1]
        file_response = client.get(
            f"/financial-data/files/{file_url}", 
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert file_response.status_code == status.HTTP_200_OK
        
    def test_admin_full_access_flow(self, client, test_admin_token):
        """Test the complete admin flow for accessing and manipulating data"""
        # Step 1: Get access limits - should show unlimited access
        limits_response = client.get(
            "/financial-data/limits", 
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert limits_response.status_code == status.HTTP_200_OK
        assert limits_response.json()["role"] == "admin"
        assert limits_response.json()["unlimited_access"] is True
        
        # Step 2: Get all financial data types
        # Get stocks
        stocks_response = client.get(
            "/financial-data?asset_type=stock", 
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert stocks_response.status_code == status.HTTP_200_OK
        assert len(stocks_response.json()["data"]) > 0
        
        # Get crypto
        crypto_response = client.get(
            "/financial-data?asset_type=crypto", 
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert crypto_response.status_code == status.HTTP_200_OK
        assert len(crypto_response.json()["data"]) > 0
        
        # Get forex
        forex_response = client.get(
            "/financial-data?asset_type=fx", 
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert forex_response.status_code == status.HTTP_200_OK
        assert len(forex_response.json()["data"]) > 0
        
        # Step 3: Download all data
        download_response = client.get(
            "/financial-data/download", 
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert download_response.status_code == status.HTTP_200_OK
        assert "download_url" in download_response.json()
        
        # Step 4: Access the download file
        file_url = download_response.json()["download_url"].split("/")[-1]
        file_response = client.get(
            f"/financial-data/files/{file_url}", 
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert file_response.status_code == status.HTTP_200_OK 