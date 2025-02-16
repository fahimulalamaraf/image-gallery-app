from fastapi import Request, HTTPException
import time
from collections import defaultdict
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        # General API rate limiting
        self.requests = defaultdict(list)  # IP -> list of timestamps
        self.max_requests = 60  # 60 requests per minute
        self.window = 60  # 1 minute window
        
        # Gemini API specific rate limiting
        self.gemini_requests = defaultdict(list)  # IP -> list of timestamps
        self.gemini_max_requests = 15  # 15 requests per minute for Gemini
        self.gemini_window = 60  # 1 minute window

    def _clean_old_requests(self, requests: List[float], window: int) -> List[float]:
        """Remove requests older than the window"""
        current_time = time.time()
        return [ts for ts in requests if current_time - ts < window]

    async def check_rate_limit(self, request: Request, is_gemini: bool = False) -> Tuple[bool, int]:
        """
        Check if request is within rate limits
        Returns: (is_allowed, wait_time_seconds)
        """
        client_ip = request.client.host
        current_time = time.time()
        
        if is_gemini:
            # Clean and update Gemini requests
            self.gemini_requests[client_ip] = self._clean_old_requests(
                self.gemini_requests[client_ip], 
                self.gemini_window
            )
            request_count = len(self.gemini_requests[client_ip])
            
            if request_count >= self.gemini_max_requests:
                # Calculate wait time
                oldest_request = min(self.gemini_requests[client_ip])
                wait_time = int(self.gemini_window - (current_time - oldest_request))
                return False, wait_time
                
            self.gemini_requests[client_ip].append(current_time)
            return True, 0
            
        else:
            # Clean and update general API requests
            self.requests[client_ip] = self._clean_old_requests(
                self.requests[client_ip], 
                self.window
            )
            request_count = len(self.requests[client_ip])
            
            if request_count >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests[client_ip])
                wait_time = int(self.window - (current_time - oldest_request))
                return False, wait_time
                
            self.requests[client_ip].append(current_time)
            return True, 0

    def get_remaining_requests(self, request: Request, is_gemini: bool = False) -> Dict:
        """Get remaining requests and reset time for a client"""
        client_ip = request.client.host
        current_time = time.time()
        
        if is_gemini:
            self.gemini_requests[client_ip] = self._clean_old_requests(
                self.gemini_requests[client_ip], 
                self.gemini_window
            )
            used_requests = len(self.gemini_requests[client_ip])
            remaining = self.gemini_max_requests - used_requests
            
            reset_time = 0 if not self.gemini_requests[client_ip] else \
                        int(self.gemini_window - (current_time - min(self.gemini_requests[client_ip])))
        else:
            self.requests[client_ip] = self._clean_old_requests(
                self.requests[client_ip], 
                self.window
            )
            used_requests = len(self.requests[client_ip])
            remaining = self.max_requests - used_requests
            
            reset_time = 0 if not self.requests[client_ip] else \
                        int(self.window - (current_time - min(self.requests[client_ip])))
        
        return {
            "remaining": remaining,
            "reset_in_seconds": reset_time,
            "limit": self.gemini_max_requests if is_gemini else self.max_requests
        } 