"""REST API for the Corporate Actions Assistant.

    pip install fastapi uvicorn
    uvicorn api:app --reload          # then open http://127.0.0.1:8000/docs

Endpoints: GET /health, POST /extract, POST /entitlement, POST /ask.
Local demo only: there is no authentication, so do not expose it to a network.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.service import NotReady, Service

service = Service()

app = FastAPI(
    title="Corporate Actions API",
    version="1.0.0",
    description=("Turns a corporate action notice into structured fields (fine-tuned model), works out an "
                 "illustrative entitlement, and answers rule questions from public SEBI documents. "
                 "Demo on synthetic notices; outputs must be verified by a person."))


class ExtractRequest(BaseModel):
    text: str = Field(min_length=30, max_length=20000, description="The text of one corporate action notice.")


class EntitlementRequest(BaseModel):
    fields: dict = Field(description="The 'fields' object returned by /extract.")
    holding: int = Field(ge=0, le=1_000_000_000, description="Shares held on the record date.")


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    k: int = Field(5, ge=1, le=10, description="How many passages to retrieve.")


@app.get("/health", summary="Which parts are ready")
def health():
    return service.status()


@app.post("/extract", summary="Notice text in, structured fields out")
def extract(req: ExtractRequest):
    try:
        return service.extract(req.text)
    except NotReady as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/entitlement", summary="Illustrative entitlement for a holding")
def entitlement(req: EntitlementRequest):
    return service.entitlement(req.fields, req.holding)


@app.post("/ask", summary="Question in, answer and source passages out")
def ask(req: AskRequest):
    try:
        return service.ask(req.question, req.k)
    except NotReady as e:
        raise HTTPException(status_code=503, detail=str(e))
