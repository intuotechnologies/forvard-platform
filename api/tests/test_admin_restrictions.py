#!/usr/bin/env python3
"""
Test script to verify admin-only endpoint restrictions in ForVARD API
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

def test_endpoint_with_auth(token: str, endpoint: str, method: str = "GET", params: Dict[str, str] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint with authentication"""
    headers = {"Authorization": f"Bearer {token}"}
    
    if data:
        headers["Content-Type"] = "application/json"
    
    try:
        if method.upper() == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
        elif method.upper() == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
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

def test_admin_endpoints():
    """Test admin-only endpoints to verify access restrictions"""
    
    print("🔐 ForVARD API Admin-Only Endpoint Tests")
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
    
    # Admin-only endpoints to test
    admin_endpoints = [
        {
            "name": "Admin Panel Root",
            "endpoint": "/admin",
            "method": "GET",
            "expected_admin": True,
            "expected_others": False,  # Should be forbidden for non-admins
        },
        {
            "name": "Admin Panel Users",
            "endpoint": "/admin/users",
            "method": "GET", 
            "expected_admin": True,
            "expected_others": False,
        },
        {
            "name": "Admin Panel Roles",
            "endpoint": "/admin/roles",
            "method": "GET",
            "expected_admin": True,
            "expected_others": False,
        }
    ]
    
    print(f"\n🧪 Testing {len(admin_endpoints)} admin endpoints for {len(tokens)} roles...")
    print("=" * 50)
    
    # Track results
    results = {}
    
    for endpoint_info in admin_endpoints:
        print(f"\n🔍 Testing: {endpoint_info['name']}")
        print(f"   Endpoint: {endpoint_info['endpoint']}")
        
        results[endpoint_info['name']] = {}
        
        for role in ["base", "senator", "admin"]:
            token = tokens[role]
            
            result = test_endpoint_with_auth(
                token, 
                endpoint_info["endpoint"], 
                endpoint_info["method"]
            )
            
            results[endpoint_info['name']][role] = result
            
            # Determine if this matches expectations
            if role == "admin":
                expected = endpoint_info["expected_admin"]
            else:
                expected = endpoint_info["expected_others"]
            
            if result["success"] == expected:
                status_icon = "✅"
                status_text = "EXPECTED"
            else:
                status_icon = "❌"
                status_text = "UNEXPECTED"
            
            print(f"   {role.upper()}: {result['status_code']} {status_icon} ({status_text})")
            
            if not result["success"] and result["status_code"]:
                if isinstance(result["data"], dict) and "detail" in result["data"]:
                    print(f"      Error: {result['data']['detail']}")
                else:
                    print(f"      Error: {result['data']}")
    
    # Test download endpoint with too many symbols (should hit limits for non-admin)
    print(f"\n🔍 Testing: Download Limits (many symbols)")
    print(f"   Endpoint: /financial-data/download")
    
    # Create a list of many symbols (more than the typical limit)
    many_symbols = ["GE", "JNJ"] * 10  # 20 symbols total
    params = {"symbols": many_symbols}
    
    for role in ["base", "senator", "admin"]:
        token = tokens[role]
        
        result = test_endpoint_with_auth(token, "/financial-data/download", "GET", params)
        
        # For admin, we expect success or at least not a 403 limit error
        # For others, we might expect a 403 limit exceeded error
        
        status_icon = "ℹ️"  # Info icon since behavior may vary
        
        print(f"   {role.upper()}: {result['status_code']} {status_icon}")
        
        if not result["success"]:
            if isinstance(result["data"], dict) and "detail" in result["data"]:
                print(f"      Error: {result['data']['detail']}")
    
    print(f"\n📊 Admin Restriction Test Summary")
    print("=" * 50)
    
    # Analyze results
    admin_protected_count = 0
    accessible_to_all_count = 0
    
    for endpoint_name, role_results in results.items():
        admin_success = role_results["admin"]["success"]
        base_success = role_results["base"]["success"]
        senator_success = role_results["senator"]["success"]
        
        if admin_success and not base_success and not senator_success:
            admin_protected_count += 1
            print(f"🔒 {endpoint_name}: ADMIN-ONLY ✅")
        elif admin_success and base_success and senator_success:
            accessible_to_all_count += 1
            print(f"🔓 {endpoint_name}: ACCESSIBLE TO ALL ⚠️")
        else:
            print(f"🤔 {endpoint_name}: MIXED ACCESS PATTERN")
    
    print(f"\n💡 FINDINGS:")
    print(f"   • Admin-protected endpoints: {admin_protected_count}")
    print(f"   • Endpoints accessible to all: {accessible_to_all_count}")
    
    if admin_protected_count > 0:
        print(f"   • ✅ The API has proper admin-only restrictions")
    else:
        print(f"   • ⚠️  No admin-only endpoints found - consider adding role-based restrictions")
    
    print(f"\n🔍 RECOMMENDATION:")
    if admin_protected_count == 0:
        print(f"   Consider implementing role-based restrictions for sensitive operations:")
        print(f"   • User management (create/delete users)")
        print(f"   • System configuration")
        print(f"   • Data export with high limits")
        print(f"   • Administrative statistics")

if __name__ == "__main__":
    try:
        test_admin_endpoints()
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1) 