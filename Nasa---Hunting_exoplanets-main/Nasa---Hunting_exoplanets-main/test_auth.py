#!/usr/bin/env python3
"""
NASA Exoplanet Classification System - Authentication Test Script
Simple test to verify the authentication system is working correctly.
"""

import requests
import sys
import time

def test_login_credentials(base_url="http://localhost:5000"):
    """Test login with correct and incorrect credentials."""
    print("🔐 Testing Authentication System")
    print("-" * 40)
    
    # Test cases: (username, password, should_succeed, description)
    test_cases = [
        ("user", "123", True, "Correct credentials"),
        ("user", "wrong", False, "Wrong password"),
        ("wrong_user", "123", False, "Wrong username"),
        ("", "", False, "Empty credentials"),
        ("admin", "admin", False, "Non-existent user"),
    ]
    
    results = []
    
    for username, password, should_succeed, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"   Username: '{username}', Password: '{password}'")
        
        try:
            session = requests.Session()
            login_data = {'username': username, 'password': password}
            response = session.post(f"{base_url}/login", data=login_data, timeout=5)
            
            success = response.status_code == 302  # Redirect indicates success
            
            if success == should_succeed:
                status = "✅ PASS"
                results.append(True)
            else:
                status = "❌ FAIL"
                results.append(False)
            
            print(f"   {status} - Expected: {should_succeed}, Got: {success}")
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR - {e}")
            results.append(False)
    
    return all(results)

def test_protected_access(base_url="http://localhost:5000"):
    """Test access to protected endpoints."""
    print("\n🔒 Testing Protected Endpoint Access")
    print("-" * 40)
    
    protected_endpoints = [
        ("/", "Main page"),
        ("/stats?dataset=k2", "Stats endpoint"),
    ]
    
    results = []
    
    for endpoint, description in protected_endpoints:
        print(f"\n🧪 Testing: {description} ({endpoint})")
        
        try:
            # Test without authentication (should redirect to login)
            response = requests.get(f"{base_url}{endpoint}", timeout=5, allow_redirects=False)
            
            if response.status_code == 302 and '/login' in response.headers.get('Location', ''):
                print("   ✅ PASS - Correctly redirects to login when not authenticated")
                results.append(True)
            else:
                print(f"   ❌ FAIL - Unexpected response: {response.status_code}")
                results.append(False)
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR - {e}")
            results.append(False)
    
    return all(results)

def test_authenticated_access(base_url="http://localhost:5000"):
    """Test access to protected endpoints with authentication."""
    print("\n🚀 Testing Authenticated Access")
    print("-" * 40)
    
    try:
        # Login first
        session = requests.Session()
        login_data = {'username': 'user', 'password': '123'}
        login_response = session.post(f"{base_url}/login", data=login_data, timeout=5)
        
        if login_response.status_code != 302:
            print("❌ Failed to login for authenticated access test")
            return False
        
        print("✅ Successfully logged in")
        
        # Test protected endpoints
        endpoints = [
            ("/", "Main page"),
            ("/stats?dataset=k2", "Stats endpoint"),
        ]
        
        results = []
        
        for endpoint, description in endpoints:
            print(f"\n🧪 Testing: {description} ({endpoint})")
            
            try:
                response = session.get(f"{base_url}{endpoint}", timeout=5)
                
                if response.status_code == 200:
                    print("   ✅ PASS - Accessible with authentication")
                    results.append(True)
                else:
                    print(f"   ❌ FAIL - Unexpected response: {response.status_code}")
                    results.append(False)
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ ERROR - {e}")
                results.append(False)
        
        # Test logout
        print(f"\n🧪 Testing: Logout")
        try:
            logout_response = session.get(f"{base_url}/logout", timeout=5, allow_redirects=False)
            
            if logout_response.status_code == 302:
                print("   ✅ PASS - Logout successful")
                results.append(True)
            else:
                print(f"   ❌ FAIL - Unexpected logout response: {logout_response.status_code}")
                results.append(False)
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR - {e}")
            results.append(False)
        
        return all(results)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication test error: {e}")
        return False

def main():
    """Run all authentication tests."""
    print("🚀 NASA Exoplanet Classification System - Authentication Test Suite")
    print("=" * 70)
    
    base_url = "http://localhost:5000"
    
    # Wait a moment for the server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Login Credentials", test_login_credentials),
        ("Protected Access", test_protected_access),
        ("Authenticated Access", test_authenticated_access),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func(base_url)
        results.append((test_name, result))
    
    print("\n" + "=" * 70)
    print("📊 Authentication Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} authentication tests passed")
    
    if passed == total:
        print("🎉 All authentication tests passed! The login system is working correctly.")
        print("\n📋 Login Credentials:")
        print("   Username: user")
        print("   Password: 123")
        return 0
    else:
        print("⚠️  Some authentication tests failed. Please check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
