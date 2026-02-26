from pydantic import BaseModel, ConfigDict


class PostGet(BaseModel):
    """Schema for post data in API responses"""
    
    id: int
    text: str
    topic: str
    
    model_config = ConfigDict(from_attributes=True)
