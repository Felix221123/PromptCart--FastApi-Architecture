from transformers import CLIPModel, CLIPProcessor
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database import get_db
import numpy as np
from src.module.models import Product, ProductEmbedding
from typing import List
from .openai_client import client
from typing import List
import torch
from PIL import Image
import requests
from io import BytesIO

router = APIRouter()
TEXT_MODEL = "text-embedding-3-large"

# Load once globally (don’t reload inside each request!)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# function to get text embedding
def get_text_embedding(text: str):
    res = client.embeddings.create(model=TEXT_MODEL, input=text)
    return res.data[0].embedding

# function to get image embedding by passing the image as a list of image_url
def get_image_embedding(image_urls: List[str]):
    """
    Generate image embeddings using CLIP from HuggingFace.
    Supports multiple images and averages their vectors.
    """
    if not image_urls:
        return None

    embeddings = []

    for url in image_urls:
        # Load image (URL or local path)
        if url.startswith("http://") or url.startswith("https://"):
            response = requests.get(url,timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(url).convert("RGB")

        # Preprocess and get embeddings
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)

        # Normalize (cosine similarity space)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        embeddings.append(image_features.cpu().numpy())

    # Average if multiple images
    return np.mean(embeddings, axis=0).squeeze().tolist()


# function router to generate embeddings
@router.post("/generate_embeddings")
def generate_embeddings(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    created, updated = 0, 0

    for product in products:
        text_input = f"{product.title}. {product.description}. Category: {product.category}"
        text_emb = get_text_embedding(text_input)

        image_emb = None
        if product.images:
            image_emb = get_image_embedding(product.images)

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


@router.get("/products")
def get_products(db: Session = Depends(get_db)):
    products = db.query(Product).all()
    return products[:2]