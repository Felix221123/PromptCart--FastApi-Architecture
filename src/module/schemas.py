# Define your module's schemas here

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

class ProductResponse(BaseModel):
    id: UUID
    title: str
    description: str
    price: float
    stock: int
    category: str
    rating: float
    warrantyInformation: str
    availabilityStatus: str
    reviews: Optional[List[Dict[str, Any]]] = None
    shippingInformation: str
    returnPolicy: str
    images: List[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ProductEmbeddingResponse(BaseModel):
    product_id: UUID
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True

