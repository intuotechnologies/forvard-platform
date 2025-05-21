import pytest
from fastapi import status

class TestAppEndpoints:
    """
    Tests for basic application endpoints
    """
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct information"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        assert "app" in response.json()
        assert "version" in response.json()
        assert "docs" in response.json()
        assert "health" in response.json()
        assert "admin" in response.json()
        
        assert response.json()["app"] == "ForVARD Financial Data API"
        assert response.json()["version"] == "1.0.0"
    
    def test_health_check_success(self, client):
        """Test the health check endpoint when all systems are working"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"
    
    def test_docs_endpoint_accessible(self, client):
        """Test the Swagger UI docs endpoint is accessible"""
        response = client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
        
    def test_redoc_endpoint_accessible(self, client):
        """Test the ReDoc docs endpoint is accessible"""
        response = client.get("/redoc")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"] 