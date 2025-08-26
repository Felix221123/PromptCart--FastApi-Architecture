# Define your module's routes here

from fastapi import APIRouter

router = APIRouter()
from . import utils, service, models


@router.get("/")
async def root():
    return {"message": "Hello World"}
