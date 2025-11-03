"""
Exception classes for Realm.ai SDK
"""

from typing import Optional


class RealmError(Exception):
    """Base exception for Realm API errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        response: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.response = response


class TimeoutError(RealmError):
    """Request timeout error"""

    def __init__(self, message: str = "Request timeout"):
        super().__init__(message, status_code=408, code="timeout")


class RateLimitError(RealmError):
    """API rate limit error"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, code="rate_limit")

