from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="AI Data Scientist",
    description="Upload a CSV; get profiling, EDA, modeling, anomaly detection, and executive summary.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "AI Data Scientist API. Use POST /api/analyze with a CSV file."}
