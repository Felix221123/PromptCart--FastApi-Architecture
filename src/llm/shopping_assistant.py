import os
import json
import re
import requests
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from typing import List, Optional, Literal, Any, Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from transformers import CLIPModel, CLIPProcessor

from src.database import get_db
from src.module.models import Product, ProductEmbedding
from src.module.schemas import ProductResponse  # you already have this
from src.utils.openai_client import client  # your OpenAI client wrapper


router = APIRouter(tags=["Shopping Assistant"])

# -----------------------
# CONFIG
# -----------------------
TEXT_MODEL = "text-embedding-3-large"

# Similarity thresholds (tune these)
MIN_DB_SCORE = float(os.getenv("MIN_DB_SCORE", "0.28"))  # combined score threshold
TOP_K = int(os.getenv("TOP_K", "8"))

# SerpAPI key MUST be env var (don’t hardcode your key)
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# Load CLIP once globally
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# -----------------------
# REQUEST/RESPONSE SCHEMAS
# -----------------------
class ImageAsset(BaseModel):
    bytes: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    public_id: Optional[str] = None
    url: str

class Message(BaseModel):
    type: Literal["user", "assistant"]
    text: str = Field(default="")
    images: Optional[List[ImageAsset]] = None


class WebProduct(BaseModel):
    title: Optional[str] = None
    price: Optional[str] = None
    extracted_price: Optional[float] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    link: Optional[str] = None
    source: Optional[str] = None
    thumbnail: Optional[str] = None


class AssistantResult(BaseModel):
    source: Literal["db", "web"]
    answer: str
    imagesUrl: List[str] = []
    products: Optional[List[Dict[str, Any]]] = None
    webResults: Optional[List[Dict[str, Any]]] = None


# -----------------------
# EMBEDDINGS
# -----------------------
def get_text_embedding(text: str) -> List[float]:
    txt = (text or "").strip()
    if not txt:
        txt = " "  # avoid OpenAI error on empty input
    res = client.embeddings.create(model=TEXT_MODEL, input=txt)
    return res.data[0].embedding


def get_image_embedding(image_urls: List[str]) -> Optional[List[float]]:
    """
    Generate image embeddings using CLIP from HuggingFace.
    Supports multiple images and averages their vectors.
    Returns a 768-dim vector.
    """
    if not image_urls:
        return None

    embeddings = []

    for url in image_urls:
        if url.startswith("http://") or url.startswith("https://"):
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(url).convert("RGB")

        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)

        # Normalise for cosine similarity space
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())

    return np.mean(embeddings, axis=0).squeeze().tolist()


def is_generic_text(text: str) -> bool:
    t = (text or "").strip().lower()
    generic_phrases = {
        "look up these items",
        "find this",
        "find these",
        "what is this",
        "help me find this",
        "help me find these",
        "search this",
        "search these",
    }
    return (t in generic_phrases) or (len(t) <= 3)


# -----------------------
# DB SEARCH (pgvector)
# -----------------------
def search_catalog(
    db: Session,
    query_text_emb: List[float],
    query_image_emb: Optional[List[float]],
    is_text_generic: bool = False,
    limit: int = TOP_K,
) -> List[Tuple[Product, float, float, float]]:
    """
    Returns list of tuples:
    (Product, score, text_similarity, image_similarity)
    """

    # Similarities (1 - cosine_distance) => higher is better
    text_sim_expr = (1 - ProductEmbedding.text_embedding.cosine_distance(query_text_emb))
    text_sim = func.coalesce(text_sim_expr, 0.0).label("text_similarity")

    if query_image_emb is not None:
        img_sim_expr = (1 - ProductEmbedding.image_embedding.cosine_distance(query_image_emb))
        img_sim = func.coalesce(img_sim_expr, 0.0).label("image_similarity")
    else:
        img_sim = func.cast(0.0, func.float).label("image_similarity")  # always 0 if no image

    # Weights: if images exist and text is generic, trust image more
    if query_image_emb is not None:
        if is_text_generic:
            w_text, w_img = 0.25, 0.75
        else:
            w_text, w_img = 0.4, 0.6
    else:
        w_text, w_img = 1.0, 0.0

    score = (w_text * text_sim + w_img * img_sim).label("score")

    q = (
        db.query(Product, score, text_sim, img_sim)
        .join(ProductEmbedding, Product.id == ProductEmbedding.product_id)
        .order_by(score.desc())
        .limit(limit)
        .all()
    )

    # Filter by minimum score
    results = [(p, float(s), float(ts), float(is_)) for (p, s, ts, is_) in q if float(s) >= MIN_DB_SCORE]
    return results


# -----------------------
# SERPAPI FALLBACK
# -----------------------
def serpapi_search_products(query: str, device: str = "mobile", max_items: int = 8) -> List[WebProduct]:
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_API_KEY is not set in environment variables.")

    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "device": device,
        "api_key": SERPAPI_KEY,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # SerpAPI varies: try immersive_products first, then shopping_results
    items = data.get("immersive_products") or data.get("shopping_results") or []
    out: List[WebProduct] = []

    for it in items[:max_items]:
        out.append(
            WebProduct(
                title=it.get("title"),
                price=it.get("price"),
                extracted_price=it.get("extracted_price"),
                rating=it.get("rating"),
                reviews=it.get("reviews") if isinstance(it.get("reviews"), int) else None,
                link=it.get("link") or it.get("product_link"),
                source=it.get("source"),
                thumbnail=it.get("thumbnail") or it.get("image"),
            )
        )

    return out


def safe_serpapi_search_products(query: str, device: str = "mobile", max_items: int = 8) -> List[WebProduct]:
    """
    Best-effort online search.
    Never raises to avoid breaking DB-backed responses.
    """
    if not SERPAPI_KEY:
        return []
    try:
        return serpapi_search_products(query=query, device=device, max_items=max_items)
    except Exception:
        return []


def clean_assistant_answer(text: str) -> str:
    """
    Force plain-text output by removing markdown formatting artifacts.
    """
    out = (text or "").strip()
    if not out:
        return "I found some options, but I could not generate a clear response."

    # Convert markdown images and links to plain text
    out = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", r"Image: \1", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", out)

    # Remove markdown emphasis/backticks/headers/blockquote markers
    out = out.replace("**", "").replace("*", "").replace("`", "")
    out = re.sub(r"^\s{0,3}#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"^\s*>\s?", "", out, flags=re.MULTILINE)

    # Keep numbered lists readable without markdown noise
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# -----------------------
# COMPOSE ANSWER (LLM)
# -----------------------
def compose_answer(
    user_message: Message,
    catalog_hits: Optional[List[Tuple[Product, float, float, float]]] = None,
    web_hits: Optional[List[WebProduct]] = None,
) -> str:
    """
    One place where we generate the final assistant answer.
    Uses OpenAI to produce a friendly response.
    """

    catalog_payload = []
    if catalog_hits:
        for p, score, text_sim, img_sim in catalog_hits:
            catalog_payload.append(
                {
                    "id": str(p.id),
                    "title": p.title,
                    "description": p.description,
                    "price": p.price,
                    "stock": p.stock,
                    "category": p.category,
                    "rating": p.rating,
                    "availabilityStatus": p.availabilityStatus,
                    "warrantyInformation": p.warrantyInformation,
                    "shippingInformation": p.shippingInformation,
                    "returnPolicy": p.returnPolicy,
                    "images": (p.images or [])[:2],
                    "score": round(score, 4),
                    "text_similarity": round(text_sim, 4),
                    "image_similarity": round(img_sim, 4),
                }
            )

    web_payload = []
    if web_hits:
        for w in web_hits:
            web_payload.append(w.model_dump())

    context = {
        "user": user_message.model_dump(),
        "catalog_results": catalog_payload,
        "web_results": web_payload,
        "rules": [
            "Prefer catalog_results if not empty.",
            "Also use web_results as additional options when available, even if catalog_results exists.",
            "Clearly label what is from our catalog and what is from online search.",
            "Be helpful, concise, and suggest 3-6 best options with short reasons.",
            "If the user provided images, mention that you matched visually as well.",
            "If you need clarification (size, budget, brand), ask 1 short follow-up question at the end.",
            "Write plain text only. Do not use markdown, asterisks, or image syntax.",
            "For each recommended option include: name, price if known, why it fits, and a link if available.",
            "Keep tone natural, human, and practical.",
        ],
    }

    system = (
        "You are PromptCart, a helpful shopping assistant. "
        "Use only the provided context data. Do not invent products or specs. "
        "Speak plainly and naturally. Output plain text only. "
        "Never use markdown like **, *, #, bullets with markdown image syntax, or code blocks."
    )

    resp = client.chat.completions.create(
        model=os.getenv("SHOP_ASSISTANT_MODEL", "gpt-4o-mini"),
        temperature=0.4,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Generate a natural plain-text shopping assistant reply from this JSON context:\n"
                           + json.dumps(context, ensure_ascii=False),
            },
        ],
    )
    raw_answer = resp.choices[0].message.content or "I found some options, but couldn’t generate a response."
    return clean_assistant_answer(raw_answer)


# -----------------------
# ONE ENDPOINT
# -----------------------
@router.post("/shop_assistant", response_model=AssistantResult)
def shop_assistant(userMessage: Message, db: Session = Depends(get_db)):
    """
    Single endpoint that:
    - takes Message {type,text,images?}
    - embeddings (text + image optional)
    - DB vector search
    - fallback to SerpAPI if no strong matches
    - returns answer + imagesUrl list
    """

    if userMessage.type != "user":
        raise HTTPException(status_code=400, detail="Message.type must be 'user'.")

    # 1) embeddings
    text_emb = get_text_embedding(userMessage.text)
    text_is_generic = is_generic_text(userMessage.text)

    image_urls = [img.url for img in (userMessage.images or []) if img.url]
    img_emb = get_image_embedding(image_urls) if image_urls else None

    # 2) DB search (combined if images exist)
    catalog_hits = search_catalog(
        db=db,
        query_text_emb=text_emb,
        query_image_emb=img_emb,
        is_text_generic=text_is_generic,
        limit=TOP_K,
    )

    # 3) Always try online search too (best-effort)
    q = userMessage.text.strip()
    if not q or text_is_generic:
        q = "product matching the uploaded image" if image_urls else "best matching product"
    web_hits = safe_serpapi_search_products(q, device="mobile", max_items=8)

    # 4) Choose primary source label: DB if present, else web
    if catalog_hits:
        answer = compose_answer(user_message=userMessage, catalog_hits=catalog_hits, web_hits=web_hits)

        imagesUrl: List[str] = []
        products_payload: List[Dict[str, Any]] = []

        for p, score, text_sim, img_sim in catalog_hits:
            # collect images
            if p.images:
                imagesUrl.extend(p.images[:2])

            # include product info + score
            prod = ProductResponse.from_orm(p).model_dump()
            prod["score"] = score
            prod["text_similarity"] = text_sim
            prod["image_similarity"] = img_sim
            products_payload.append(prod)

        # de-duplicate image urls while preserving order
        if web_hits:
            imagesUrl.extend([w.thumbnail for w in web_hits if w.thumbnail])
        seen = set()
        imagesUrl = [u for u in imagesUrl if not (u in seen or seen.add(u))]

        return AssistantResult(
            source="db",
            answer=answer,
            imagesUrl=imagesUrl,
            products=products_payload,
            webResults=[w.model_dump() for w in web_hits] if web_hits else None,
        )

    # 5) Web-only fallback if DB has no strong matches
    answer = compose_answer(user_message=userMessage, catalog_hits=None, web_hits=web_hits)

    imagesUrl = [w.thumbnail for w in web_hits if w.thumbnail]
    seen = set()
    imagesUrl = [u for u in imagesUrl if not (u in seen or seen.add(u))]

    return AssistantResult(
        source="web",
        answer=answer,
        imagesUrl=imagesUrl,
        products=None,
        webResults=[w.model_dump() for w in web_hits],
    )
