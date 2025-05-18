import pytest
from fastapi import status
import json
import os

class TestFinancialDataEndpoints:
    """
    Tests for financial data endpoints
    """
    
    def test_get_financial_data_no_auth(self, client):
        """Test accessing financial data without authentication"""
        response = client.get("/financial-data")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "detail" in response.json()
    
    def test_get_financial_data_admin(self, client, test_admin_token):
        """Test admin can access all financial data"""
        response = client.get(
            "/financial-data",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "data" in response.json()
        assert "total" in response.json()
        assert response.json()["total"] > 0
        assert len(response.json()["data"]) > 0
    
    def test_get_financial_data_with_filters(self, client, test_admin_token):
        """Test financial data filtering"""
        response = client.get(
            "/financial-data?symbol=AAPL&asset_type=stock",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "data" in response.json()
        assert len(response.json()["data"]) > 0
        assert all(item["symbol"] == "AAPL" for item in response.json()["data"])
        assert all(item["asset_type"] == "stock" for item in response.json()["data"])
    
    def test_get_financial_data_with_date_filters(self, client, test_admin_token):
        """Test financial data date filtering"""
        response = client.get(
            "/financial-data?start_date=2023-01-02&end_date=2023-01-02",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "data" in response.json()
        assert len(response.json()["data"]) > 0
        assert all(item["observation_date"] == "2023-01-02" for item in response.json()["data"])
    
    def test_get_financial_data_with_fields(self, client, test_admin_token):
        """Test financial data field selection"""
        response = client.get(
            "/financial-data?fields=pv&fields=gk",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "data" in response.json()
        assert len(response.json()["data"]) > 0
        assert "pv" in response.json()["data"][0]
        assert "gk" in response.json()["data"][0]
    
    def test_get_financial_data_base_role_limits(self, client, test_base_token):
        """Test base role access limits are enforced"""
        # First get data for an asset type that has limits
        response = client.get(
            "/financial-data?asset_type=stock",
            headers={"Authorization": f"Bearer {test_base_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "data" in response.json()
        
        # Count unique symbols in response
        symbols = set(item["symbol"] for item in response.json()["data"])
        assert len(symbols) <= 10  # Base role has max 10 stocks limit
    
    def test_get_financial_data_senator_role_limits(self, client, test_senator_token):
        """Test senator role access limits are enforced"""
        response = client.get(
            "/financial-data?asset_type=crypto",
            headers={"Authorization": f"Bearer {test_senator_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        symbols = set(item["symbol"] for item in response.json()["data"])
        assert len(symbols) <= 30  # Senator role has max 30 cryptos limit
    
    def test_get_access_limits_admin(self, client, test_admin_token):
        """Test admin can see access limits"""
        response = client.get(
            "/financial-data/limits",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "limits" in response.json()
        assert response.json()["role"] == "admin"
        assert response.json()["unlimited_access"] is True
    
    def test_get_access_limits_base(self, client, test_base_token):
        """Test base user can see their access limits"""
        response = client.get(
            "/financial-data/limits",
            headers={"Authorization": f"Bearer {test_base_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "limits" in response.json()
        assert response.json()["role"] == "base"
        assert response.json()["unlimited_access"] is False
        assert "stock" in response.json()["limits"]
        assert response.json()["limits"]["stock"] == 10
    
    def test_download_financial_data(self, client, test_admin_token):
        """Test downloading financial data as CSV"""
        response = client.get(
            "/financial-data/download?asset_type=stock",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "download_url" in response.json()
        assert "filename" in response.json()
        
        # Verify the file is accessible
        filename = response.json()["filename"]
        file_response = client.get(
            f"/financial-data/files/{filename}",
            headers={"Authorization": f"Bearer {test_admin_token}"}
        )
        
        assert file_response.status_code == status.HTTP_200_OK
        assert file_response.headers["content-type"] == "text/csv"
    
    def test_download_with_limited_access(self, client, test_base_token):
        """Test downloading financial data with limited access"""
        # Try to download more symbols than allowed
        response = client.get(
            "/financial-data/download?symbols=AAPL&symbols=MSFT&symbols=GOOGL&symbols=AMZN&symbols=META&symbols=NFLX&symbols=TSLA&symbols=NVDA&symbols=ADBE&symbols=INTC&symbols=CSCO",
            headers={"Authorization": f"Bearer {test_base_token}"}
        )
        
        # Should fail as base users can only access 10 stocks
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "detail" in response.json()
        assert "exceeds your access limit" in response.json()["detail"] 