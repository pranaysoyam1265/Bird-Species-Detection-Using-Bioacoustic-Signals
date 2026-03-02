"""
BirdSense Backend — Species Router
GET /species      — list all 87 species
GET /species/{id} — single species by index
"""

from fastapi import APIRouter, HTTPException

from models.schemas import SpeciesItem, SpeciesListResponse
from services.inference import get_all_species, get_species_by_index

router = APIRouter()


@router.get(
    "/species",
    response_model=SpeciesListResponse,
    summary="List all detectable bird species",
)
async def list_species():
    """Return all 87 bird species the model can identify."""
    species = get_all_species()
    return SpeciesListResponse(
        count=len(species),
        species=[SpeciesItem(**sp) for sp in species],
    )


@router.get(
    "/species/{index}",
    response_model=SpeciesItem,
    summary="Get species by model index",
)
async def get_species(index: int):
    """Return a single species by its model class index (0–86)."""
    sp = get_species_by_index(index)
    if sp is None:
        raise HTTPException(status_code=404, detail=f"No species with index {index}")
    return SpeciesItem(**sp)
