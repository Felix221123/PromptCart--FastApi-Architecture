# Define your module's crud operations here

from typing import List
from fastapi import Depends
from sqlalchemy.orm import Session
from . import models, schemas
from src.database import get_db
from .models import Product, ProductEmbedding

async def get_all_products(db: Session = Depends(get_db)) -> List[Product]:
    pass
