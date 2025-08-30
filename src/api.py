# Define all the router here.

from fastapi import APIRouter
from .module.views import router as module_router
from .utils.embedding import router as utils_router
from .utils.product_search import router as search_router

router = APIRouter()
router.include_router(module_router)
router.include_router(utils_router)
router.include_router(search_router)


