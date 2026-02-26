from pydantic import BaseModel, ConfigDict


class PostGet(BaseModel):
    """
    Schema for post data in API responses.
    
    Attributes:
        id: Unique post identifier
        text: Post content text
        topic: Post topic/category
    """
    id: int
    text: str
    topic: str
    
    model_config = ConfigDict(from_attributes=True)
