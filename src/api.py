# Define all the router here.

from fastapi import APIRouter
from .module.views import router as module_router

router = APIRouter()
router.include_router(module_router)
