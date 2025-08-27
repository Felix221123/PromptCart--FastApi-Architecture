from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database import get_db
import numpy as np
from src.module.models import Product, ProductEmbedding
from typing import List
import openai

router = APIRouter()
TEXT_MODEL = "text-embedding-3-large"
IMAGE_MODEL = "gpt-4o-mini"


# function to get text embedding
def get_text_embedding(text: str):
    res = openai.embeddings.create(model=TEXT_MODEL, input=text)
    return res.data[0].embedding

# function to get image embedding by passing the image as a list of image_url
def get_image_embedding(image_urls: List[str]):
    if len(image_urls) < 1:
        return None

    embeddings = []

    for url in image_urls:
        res = openai.embeddings.create(model=IMAGE_MODEL, input=url)
        embeddings.append(res.data[0].embedding)

    # average across all image vectors
    return np.mean(embeddings, axis=0).tolist()


@router.post("/generate_embeddings")
def generate_embeddings(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    created, updated = 0, 0

    for product in products:
        text_input = f"{product.title}. {product.description}. Category: {product.category}"
        text_emb = get_text_embedding(text_input)

        image_emb = None
        if product.images:
            image_emb = get_image_embedding(product.images[0])

        embedding = db.query(ProductEmbedding).filter_by(product_id=product.id).first()
        if embedding:
            embedding.text_embedding = text_emb
            embedding.image_embedding = image_emb
            updated += 1
        else:
            new_emb = ProductEmbedding(
                product_id=product.id,
                text_embedding=text_emb,
                image_embedding=image_emb
            )
            db.add(new_emb)
            created += 1

    db.commit()
    return {"created": created, "updated": updated, "total": len(products)}