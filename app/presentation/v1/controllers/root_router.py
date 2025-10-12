
from fastapi import APIRouter
from app.presentation.v1.controllers.placement import router as placement_router

root_router = APIRouter(prefix="/api/v1")

root_router.include_router(placement_router)
