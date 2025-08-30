# product search
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from src.database import get_db
from .openai_client import client
from ..module import models
from ..module.schemas import ProductResponse
from typing import List

router = APIRouter()


# function to search for products
@router.post("/product_search", response_model=None)
def search_similar_items_by_text(query_text: str, db: Session = Depends(get_db)):
    """
    Searches for items with embeddings similar to the embedding of the query text.
    """
    try:
        # generate an embedding for the query via openai api
        response = client.embeddings.create(
            input=query_text.strip(),
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    # Perform similarity search (join ProductEmbedding → Product)
    query_response = (
        db.query(models.Product)
        .join(models.ProductEmbedding, models.Product.id == models.ProductEmbedding.product_id)
        .order_by(models.ProductEmbedding.text_embedding.cosine_distance(query_embedding))
        .limit(10)
        .all()
    )

    if not query_response:
        raise HTTPException(status_code=404, detail="No similar items found")

    return {"products": [ProductResponse.from_orm(p) for p in query_response]}

