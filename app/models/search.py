from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    filters: Optional[Dict[str, Any]] = None

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip() 