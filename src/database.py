# Define database connection here

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src import constants
from sqlalchemy.ext.declarative import declarative_base


engine = create_engine(
    constants.config["DATABASE_URL"], connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
