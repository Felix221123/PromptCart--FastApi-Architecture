# Define your module's models here

from pydantic import BaseModel
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid


Base = declarative_base()

# ===========================
# Product Model
# ===========================
class Product(Base):
    __tablename__ = "products"
    __table_args__ = {"schema": "public"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, nullable=False)
    category = Column(String, nullable=False)
    rating = Column(Float, nullable=False)
    warrantyInformation = Column(String, nullable=False)
    availabilityStatus = Column(String, nullable=False)
    reviews = Column(JSON, nullable=True)
    shippingInformation = Column(String, nullable=False)
    returnPolicy = Column(String, nullable=False)
    images = Column(ARRAY(Text), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    embedding = relationship("ProductEmbedding", back_populates="product", uselist=False)


# ===========================
# Product Embedding Model (pgvector!)
# ===========================
class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"
    __table_args__ = {"schema": "public"}

    product_id = Column(UUID(as_uuid=True), ForeignKey("public.products.id", ondelete="CASCADE"), primary_key=True)
    text_embedding = Column(Vector(3072), nullable=True)
    image_embedding = Column(Vector(768), nullable=True)

    product = relationship("Product", back_populates="embedding")