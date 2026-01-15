# BroCode Sentinel Python SDK
# Official Python client for BroCode Sentinel API

import requests
import json
import time
from typing import Dict, List, Optional, Any

class BroCodeSDK:
    """
    Official Python SDK for BroCode Sentinel API.
    
    Usage:
        from brocode_sdk import BroCodeSDK
        
        client = BroCodeSDK(
            api_key="bk_your_api_key",
            base_url="https://your-domain.com"
        )
        
        # Register user
        result = client.register_user("john", "english", keystrokes_data)
        
        # Verify user
        result = client.verify_user("john", "english", keystrokes_data)
        
        # Get trust score
        score = client.get_trust_score("john")
    """
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:5000", timeout: int = 30):
        """
        Initialize BroCode SDK client
        
        Args:
            api_key: Your API key (starts with 'bk_')
            base_url: Base URL of BroCode Sentinel API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request to API"""
        url = f"{self.base_url}/api/v1{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                error_data = {"error": str(e)}
            
            raise BroCodeAPIError(
                message=error_data.get("error", "API request failed"),
                status_code=e.response.status_code,
                details=error_data
            )
        except requests.exceptions.RequestException as e:
            raise BroCodeAPIError(
                message=f"Request failed: {str(e)}",
                status_code=0,
                details={}
            )
    
    def health_check(self) -> Dict:
        """
        Check API health status
        
        Returns:
            Dict with health status
        """
        return self._request('GET', '/health')
    
    def register_user(self, username: str, language: str, keystrokes: List[Dict], email: Optional[str] = None) -> Dict:
        """
        Register a new user
        
        Args:
            username: Username
            language: Language (english, spanish, etc.)
            keystrokes: List of keystroke timing data
            email: Optional email address
        
        Returns:
            Registration result
        """
        data = {
            "username": username,
            "language": language,
            "keystrokes": keystrokes
        }
        
        if email:
            data["email"] = email
        
        return self._request('POST', '/auth/register', data=data)
    
    def verify_user(self, username: str, language: str, keystrokes: List[Dict]) -> Dict:
        """
        Verify user authentication
        
        Args:
            username: Username
            language: Language
            keystrokes: Keystroke timing data
        
        Returns:
            Verification result with trust score
        """
        data = {
            "username": username,
            "language": language,
            "keystrokes": keystrokes
        }
        
        return self._request('POST', '/auth/verify', data=data)
    
    def get_trust_score(self, username: str) -> Dict:
        """
        Get current trust score for user
        
        Args:
            username: Username
        
        Returns:
            Trust score and components
        """
        return self._request('GET', f'/users/{username}/trust')
    
    def get_user_sessions(self, username: str, limit: int = 10, offset: int = 0) -> Dict:
        """
        Get user's session history
        
        Args:
            username: Username
            limit: Number of sessions to return
            offset: Pagination offset
        
        Returns:
            List of sessions
        """
        params = {"limit": limit, "offset": offset}
        return self._request('GET', f'/users/{username}/sessions', params=params)
    
    def register_webhook(self, url: str, events: List[str], secret: Optional[str] = None) -> Dict:
        """
        Register a webhook for events
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional webhook secret
        
        Returns:
            Webhook registration result
        """
        data = {
            "url": url,
            "events": events
        }
        
        if secret:
            data["secret"] = secret
        
        return self._request('POST', '/webhooks', data=data)
    
    def get_stats(self) -> Dict:
        """
        Get system statistics
        
        Returns:
            System statistics
        """
        return self._request('GET', '/stats')


class BroCodeAPIError(Exception):
    """Exception raised for API errors"""
    
    def __init__(self, message: str, status_code: int, details: Dict):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)
    
    def __str__(self):
        return f"BroCodeAPIError({self.status_code}): {self.message}"


# Convenience functions

def create_client(api_key: str, base_url: str = "http://localhost:5000") -> BroCodeSDK:
    """
    Create a BroCode SDK client
    
    Args:
        api_key: Your API key
        base_url: Base URL of API
    
    Returns:
        BroCodeSDK client instance
    """
    return BroCodeSDK(api_key=api_key, base_url=base_url)


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = BroCodeSDK(
        api_key="bk_your_api_key_here",
        base_url="http://localhost:5000"
    )
    
    # Health check
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
    except BroCodeAPIError as e:
        print(f"Error: {e}")
    
    # Register user
    try:
        keystrokes = [
            {"key": "a", "press_time": 100, "release_time": 150},
            {"key": "b", "press_time": 200, "release_time": 250}
        ]
        
        result = client.register_user(
            username="john_doe",
            language="english",
            keystrokes=keystrokes,
            email="john@example.com"
        )
        
        print(f"Registration: {result}")
    except BroCodeAPIError as e:
        print(f"Error: {e}")
    
    # Get trust score
    try:
        score = client.get_trust_score("john_doe")
        print(f"Trust Score: {score['trust_score']}")
    except BroCodeAPIError as e:
        print(f"Error: {e}")
