import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for sub in ("features", "models", "feedback"):
    sys.path.insert(0, str(ROOT / sub))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import correlations, recommendations, risk, trend

app = FastAPI(title="Grade Change Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (trend.router, risk.router, correlations.router, recommendations.router):
    app.include_router(router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}