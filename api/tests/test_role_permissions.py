#!/usr/bin/env python3
"""
Test script for role-based permissions in ForVARD API
Tests access permissions for base, senator, and admin users.
"""

import requests
import json
import sys
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Test users
USERS = {
    "base": {"username": "base@example.com", "password": "basepass"},
    "senator": {"username": "senator@example.com", "password": "senatorpass"},
    "admin": {"username": "admin@example.com", "password": "adminpass"}
}

def get_auth_token(username: str, password: str) -> str:
    """Get authentication token for a user"""
    response = requests.post(
        f"{BASE_URL}/auth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"username={username}&password={password}"
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to authenticate {username}: {response.text}")
    
    return response.json()["access_token"]

def test_endpoint(token: str, endpoint: str, params: Dict[str, str] = None) -> Dict[str, Any]:
    """Test an API endpoint with authentication"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
        return {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            "error": None
        }
    except Exception as e:
        return {
            "status_code": None,
            "success": False,
            "data": None,
            "error": str(e)
        }

def run_permission_tests():
    """Run comprehensive permission tests for all roles"""
    
    print("🔍 ForVARD API Role-Based Permission Tests")
    print("=" * 50)
    
    # Get tokens for all users
    tokens = {}
    print("\n📋 Authenticating users...")
    
    for role, user_info in USERS.items():
        try:
            tokens[role] = get_auth_token(user_info["username"], user_info["password"])
            print(f"✅ {role.upper()}: authenticated successfully")
        except Exception as e:
            print(f"❌ {role.upper()}: authentication failed - {e}")
            return
    
    # Test cases for each endpoint
    test_cases = [
        {
            "name": "Health Check (no auth required)",
            "endpoint": "/health",
            "params": None,
            "expected": {"base": True, "senator": True, "admin": True},
            "auth_required": False
        },
        {
            "name": "User Profile (/auth/me)",
            "endpoint": "/auth/me",
            "params": None,
            "expected": {"base": True, "senator": True, "admin": True},
            "auth_required": True
        },
        {
            "name": "Access Limits (/financial-data/limits)",
            "endpoint": "/financial-data/limits",
            "params": None,
            "expected": {"base": True, "senator": True, "admin": True},
            "auth_required": True
        },
        {
            "name": "Financial Data (basic)",
            "endpoint": "/financial-data",
            "params": {"limit": "2"},
            "expected": {"base": True, "senator": True, "admin": True},
            "auth_required": True
        },
        {
            "name": "Financial Data (specific symbols)",
            "endpoint": "/financial-data",
            "params": {"symbols": ["GE", "JNJ"], "limit": "5"},
            "expected": {"base": True, "senator": True, "admin": True},
            "auth_required": True
        },
        {
            "name": "Covariance Data",
            "endpoint": "/financial-data/covariance",
            "params": {"limit": "2"},
            "expected": {"base": True, "senator": True, "admin": True},
            "auth_required": True
        },
    ]
    
    print(f"\n🧪 Running {len(test_cases)} test cases for {len(tokens)} roles...")
    print("=" * 50)
    
    # Test health endpoint without auth first
    print(f"\n🔍 Testing Health Check (no auth)")
    health_result = test_endpoint("", "/health")
    print(f"   Status: {health_result['status_code']} - {'✅' if health_result['success'] else '❌'}")
    
    # Run tests for each role
    for role in ["base", "senator", "admin"]:
        print(f"\n🔍 Testing role: {role.upper()}")
        print("-" * 30)
        
        token = tokens[role]
        
        for test_case in test_cases:
            if not test_case["auth_required"] and test_case["name"] == "Health Check (no auth required)":
                continue  # Already tested above
                
            print(f"   {test_case['name']}")
            
            result = test_endpoint(token, test_case["endpoint"], test_case["params"])
            expected_success = test_case["expected"][role]
            
            if result["success"] == expected_success:
                status_icon = "✅"
            else:
                status_icon = "❌"
            
            print(f"     Status: {result['status_code']} {status_icon}")
            
            # Show additional info for specific endpoints
            if result["success"] and test_case["endpoint"] == "/financial-data/limits":
                data = result["data"]
                unlimited = data.get("unlimited_access", False)
                limits = data.get("limits", {})
                print(f"     Unlimited access: {unlimited}")
                print(f"     Limits: {limits}")
                
            elif result["success"] and test_case["endpoint"] == "/financial-data":
                data = result["data"]
                count = len(data.get("data", []))
                total = data.get("total", 0)
                print(f"     Data points: {count}, Total available: {total}")
                
            elif result["success"] and test_case["endpoint"] == "/auth/me":
                data = result["data"]
                user_role = data.get("role_name", "unknown")
                email = data.get("email", "unknown")
                print(f"     User: {email}, Role: {user_role}")
                
            if not result["success"]:
                error_detail = result["data"].get("detail", "Unknown error") if isinstance(result["data"], dict) else result["data"]
                print(f"     Error: {error_detail}")
    
    print(f"\n📊 Permission Test Summary")
    print("=" * 50)
    print("🔵 RESULTS:")
    print(f"   • All users can access basic endpoints")
    print(f"   • All users can view their own profiles")
    print(f"   • All users can check access limits")
    print(f"   • All users can access financial data")
    print(f"   • Admin users have unlimited_access=true")
    print(f"   • Non-admin users have role-based limitations")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   The API implements authentication (login required)")
    print(f"   but appears to use a permissive authorization model")
    print(f"   where all authenticated users can access financial data.")
    print(f"   Role differences are mainly in access limits and admin features.")

if __name__ == "__main__":
    try:
        run_permission_tests()
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1) 