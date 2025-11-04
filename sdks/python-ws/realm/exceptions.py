"""
Exception classes for Realm WebSocket client
"""


class RealmError(Exception):
    """Base exception for Realm errors"""
    pass


class ConnectionError(RealmError):
    """Connection-related errors"""
    pass


class AuthenticationError(RealmError):
    """Authentication errors"""
    pass


class RateLimitError(RealmError):
    """Rate limit exceeded"""
    
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutError(RealmError):
    """Request timeout"""
    pass


class FunctionError(RealmError):
    """Function execution error"""
    
    def __init__(self, message: str, code: str = None):
        super().__init__(message)
        self.code = code

