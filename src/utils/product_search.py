# product search
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from src.database import get_db
from .openai_client import client
from ..module import models
from ..module.schemas import ProductResponse
from typing import List

router = APIRouter()

# similarity threshold for search results
SIMILARITY_THRESHOLD = 0.2

# function to search for products
@router.post("/product_search", response_model=None)
def search_products(query_text: str, db: Session = Depends(get_db)):
    """
    Searches for items with embeddings similar to the embedding of the query text.
    """
    try:
        # Generate an embedding for the query via OpenAI API
        response = client.embeddings.create(
            input=query_text.strip(),
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    # Perform similarity search with threshold filter
    query_response = (
        db.query(models.Product, 1 - models.ProductEmbedding.text_embedding.cosine_distance(query_embedding).label("similarity"))
        .join(models.ProductEmbedding, models.Product.id == models.ProductEmbedding.product_id)
        .order_by(models.ProductEmbedding.text_embedding.cosine_distance(query_embedding))
        .limit(20)
        .all()
    )

    if not query_response:
        raise HTTPException(status_code=404, detail="No similar items found")

    # Filter out low-similarity results
    filtered_products = [
        product for product, similarity in query_response if similarity >= SIMILARITY_THRESHOLD
    ]

    if not filtered_products:
        raise HTTPException(status_code=404, detail="No relevant results found")

    return {"products": [ProductResponse.from_orm(p) for p in filtered_products]}