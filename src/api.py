# Define all the router here.

from fastapi import APIRouter
from .module.views import router as module_router
from .utils.embedding import router as utils_router
from .utils.product_search import router as search_router
from .llm.shopping_assistant import router as shopping_router


router = APIRouter()
router.include_router(module_router)
router.include_router(utils_router)
router.include_router(search_router)
router.include_router(shopping_router)

[
    {
        "url": "https://res.cloudinary.com/drjjavzx5/image/upload/v1771519171/promptcart/txg6ndyl94sk8favbqd6.png",
        "public_id": "promptcart/txg6ndyl94sk8favbqd6",
        "width": 2278,
        "height": 546,
        "bytes": 423502
    },
    {
        "url": "https://res.cloudinary.com/drjjavzx5/image/upload/v1771519175/promptcart/zocxmt0z2etrjuvz482n.png",
        "public_id": "promptcart/zocxmt0z2etrjuvz482n",
        "width": 1818,
        "height": 1204,
        "bytes": 1125255
    }
]
