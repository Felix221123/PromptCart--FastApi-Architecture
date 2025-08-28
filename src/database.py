# Define database connection here

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src import constants

engine = create_engine(
    constants.config["DATABASE_URL"], echo=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
