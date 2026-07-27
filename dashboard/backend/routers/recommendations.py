"""Recommendation and feedback endpoints.

GET  /api/recommendations?status=pending   filtered suggestion list
POST /api/recommendations/{id}/decision    accept / reject
GET  /api/accuracy                         feedback accuracy stats
"""

import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RECOMMENDATIONS_PATH = ROOT / "models" / "artifacts" / "recommendations.json"

# feedback_store is importable thanks to main.py's sys.path setup
import feedback_store

router = APIRouter()

_suggestions = None


def _load_suggestions():
    global _suggestions
    if _suggestions is None:
        if not RECOMMENDATIONS_PATH.exists():
            raise HTTPException(status_code=404, detail="recommendations.json not found. Run setpoint_recommender.py first.")
        with open(RECOMMENDATIONS_PATH) as f:
            _suggestions = json.load(f)
    return _suggestions


class DecisionBody(BaseModel):
    decision: str


@router.get("/recommendations")
def get_recommendations(status: str = "pending"):
    suggestions = _load_suggestions()
    if status == "all":
        return suggestions
    return [s for s in suggestions if s.get("status", "pending") == status]


@router.post("/recommendations/{suggestion_id}/decision")
def post_decision(suggestion_id: str, body: DecisionBody):
    if body.decision not in ("accepted", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'accepted' or 'rejected'")

    # Update the in-memory list
    suggestions = _load_suggestions()
    found = False
    for s in suggestions:
        if s["suggestion_id"] == suggestion_id:
            s["status"] = body.decision
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"suggestion_id {suggestion_id} not found")

    # Also record in feedback store if it has the suggestion seeded
    try:
        feedback_store.record_decision(suggestion_id, body.decision)
    except (KeyError, Exception):
        pass  # feedback log may not have this suggestion seeded yet

    return {"suggestion_id": suggestion_id, "decision": body.decision}


@router.get("/accuracy")
def get_accuracy():
    try:
        return feedback_store.compute_accuracy()
    except Exception:
        return {"n_evaluated": 0, "accuracy": None}
