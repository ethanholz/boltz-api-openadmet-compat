from __future__ import annotations

import os
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

MODEL = "adme-v1"
VERSION = os.getenv("ADME_MODEL_VERSION", "openadmet-models")
WORKSPACE = os.getenv("ADME_WORKSPACE_ID", "default")
COST_PER_MOLECULE = Decimal(os.getenv("ADME_COST_PER_MOLECULE_USD", "0.0500"))
DISCLAIMER = (
    "This is an estimate only and may differ from your actual charges. "
    "Final billing is based on the configured deployment pricing."
)


def now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputMolecule(StrictModel):
    smiles: str = Field(min_length=1)
    id: str | None = None


class Input(StrictModel):
    molecules: list[InputMolecule] = Field(min_length=1, max_length=128)


class PredictionRequest(StrictModel):
    input: Input
    model: Literal["adme-v1"]
    idempotency_key: str | None = None
    workspace_id: str | None = None


class Error(BaseModel):
    code: str
    message: str
    details: dict | None = None


class Adme(BaseModel):
    lipophilicity: float
    permeability: float
    solubility: Literal["high-confidence", "medium-confidence", "high-risk"]


class OutputMolecule(BaseModel):
    id: str
    adme: Adme | None
    error: Error | None
    smiles: str
    status: Literal["succeeded", "failed"]
    external_id: str | None = None


class Output(BaseModel):
    molecules: list[OutputMolecule]


class Prediction(BaseModel):
    id: str
    completed_at: datetime | None = None
    created_at: datetime
    data_deleted_at: datetime | None = None
    error: Error | None = None
    expires_at: datetime | None = None
    input: Input | None
    livemode: bool = False
    model: Literal["adme-v1"] = MODEL
    output: Output | None = None
    started_at: datetime | None = None
    status: Literal["pending", "running", "succeeded", "failed"]
    version: str = VERSION
    workspace_id: str
    idempotency_key: str | None = None


class PredictionSummary(BaseModel):
    id: str
    completed_at: datetime | None
    created_at: datetime
    data_deleted_at: datetime | None
    error: Error | None
    expires_at: datetime | None
    livemode: bool
    model: Literal["adme-v1"]
    started_at: datetime | None
    status: Literal["pending", "running", "succeeded", "failed"]
    version: str
    workspace_id: str
    idempotency_key: str | None


class PredictionPage(BaseModel):
    data: list[PredictionSummary]
    first_id: str | None
    has_more: bool
    last_id: str | None


class DeleteResponse(BaseModel):
    id: str
    data_deleted: Literal[True] = True
    data_deleted_at: datetime


class Breakdown(BaseModel):
    application: Literal["adme"] = "adme"
    cost_per_unit_usd: str
    num_units: int


class CostResponse(BaseModel):
    breakdown: Breakdown
    disclaimer: str = DISCLAIMER
    estimated_cost_usd: str


PredictionValues = tuple[float, float, Literal["high-confidence", "medium-confidence", "high-risk"]]
Predictor = Callable[[list[str]], list[PredictionValues | Error]]


class OpenAdmetPredictor:
    """Adapter for trained Anvil models mounted into the OpenADMET image."""

    def __init__(self) -> None:
        dirs = os.getenv("ADME_MODEL_DIRS", "")
        self.model_dirs = [item for item in dirs.split(os.pathsep) if item]
        self.columns = {
            "lipophilicity": os.getenv("ADME_LIPOPHILICITY_COLUMN", ""),
            "permeability": os.getenv("ADME_PERMEABILITY_COLUMN", ""),
            "solubility": os.getenv("ADME_SOLUBILITY_COLUMN", ""),
        }

    def __call__(self, smiles: list[str]) -> list[PredictionValues | Error]:
        if not self.model_dirs or not all(self.columns.values()):
            raise RuntimeError(
                "Configure ADME_MODEL_DIRS and the three ADME_*_COLUMN variables"
            )
        import pandas as pd
        from openadmet.models.inference.inference import predict

        frame = predict(
            pd.DataFrame({"smiles": smiles}),
            "smiles",
            self.model_dirs,
            accelerator=os.getenv("ADME_ACCELERATOR", "cpu"),
            log=False,
        )
        results: list[PredictionValues | Error] = []
        solubility_mode = os.getenv("ADME_SOLUBILITY_MODE", "score")
        low = float(os.getenv("ADME_SOLUBILITY_HIGH_RISK_MAX", "0.33"))
        high = float(os.getenv("ADME_SOLUBILITY_HIGH_CONFIDENCE_MIN", "0.67"))
        logd_confident = float(os.getenv("ADME_SOLUBILITY_HIGH_CONFIDENCE_MAX", "1"))
        logd_risky = float(os.getenv("ADME_SOLUBILITY_HIGH_RISK_MIN", "3"))
        for index in range(len(smiles)):
            values = [frame.at[index, column] for column in self.columns.values()]
            if any(pd.isna(value) for value in values):
                results.append(Error(code="invalid_smiles", message="SMILES could not be featurized"))
                continue
            solubility_value = values[2]
            if solubility_value in {"high-confidence", "medium-confidence", "high-risk"}:
                solubility = solubility_value
            elif solubility_mode == "logd":
                solubility = (
                    "high-confidence" if float(solubility_value) <= logd_confident
                    else "high-risk" if float(solubility_value) >= logd_risky
                    else "medium-confidence"
                )
            else:
                solubility = (
                    "high-risk" if float(solubility_value) < low
                    else "high-confidence" if float(solubility_value) >= high
                    else "medium-confidence"
                )
            results.append((float(values[0]), float(values[1]), solubility))
        return results


class Store:
    def __init__(self) -> None:
        self.items: dict[str, Prediction] = {}
        self.keys: dict[tuple[str, str], str] = {}
        self.lock = threading.Lock()


app = FastAPI(title="ADME API", version="1.0.0")
app.state.store = Store()
app.state.predictor = OpenAdmetPredictor()


def workspace(requested: str | None) -> str:
    return requested or WORKSPACE


def get_prediction(prediction_id: str, workspace_id: str | None = None) -> Prediction:
    item = app.state.store.items.get(prediction_id)
    if item is None or (workspace_id is not None and item.workspace_id != workspace_id):
        raise HTTPException(404, "ADME prediction not found")
    return item


@app.post("/compute/v1/predictions/adme", response_model=Prediction)
def start(request: PredictionRequest) -> Prediction:
    target = workspace(request.workspace_id)
    store = app.state.store
    with store.lock:
        if request.idempotency_key:
            existing = store.keys.get((target, request.idempotency_key))
            if existing:
                return store.items[existing]
        timestamp = now()
        item = Prediction(
            id=str(uuid4()), created_at=timestamp, input=request.input,
            started_at=timestamp, status="running", workspace_id=target,
            idempotency_key=request.idempotency_key,
        )
        store.items[item.id] = item
        if item.idempotency_key:
            store.keys[(target, item.idempotency_key)] = item.id
    try:
        values = app.state.predictor([m.smiles for m in request.input.molecules])
        if len(values) != len(request.input.molecules):
            raise RuntimeError("predictor returned the wrong number of results")
        molecules = []
        for molecule, result in zip(request.input.molecules, values, strict=True):
            molecule_id = str(uuid4())
            if isinstance(result, Error):
                molecules.append(OutputMolecule(id=molecule_id, adme=None, error=result, smiles=molecule.smiles, status="failed", external_id=molecule.id))
            else:
                molecules.append(OutputMolecule(id=molecule_id, adme=Adme(lipophilicity=result[0], permeability=result[1], solubility=result[2]), error=None, smiles=molecule.smiles, status="succeeded", external_id=molecule.id))
        item.output = Output(molecules=molecules)
        item.status = "succeeded"
    except Exception as exc:
        item.error = Error(code="prediction_failed", message=str(exc))
        item.status = "failed"
    item.completed_at = now()
    return item


@app.get("/compute/v1/predictions/adme", response_model=PredictionPage)
def list_predictions(
    after_id: str | None = None,
    before_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    workspace_id: str | None = None,
) -> PredictionPage:
    if after_id and before_id:
        raise HTTPException(400, "after_id and before_id are mutually exclusive")
    items = [item for item in app.state.store.items.values() if item.workspace_id == workspace(workspace_id)]
    ids = [item.id for item in items]
    if after_id:
        if after_id not in ids:
            raise HTTPException(400, "after_id is not in this workspace")
        items = items[ids.index(after_id) + 1 :]
    elif before_id:
        if before_id not in ids:
            raise HTTPException(400, "before_id is not in this workspace")
        items = items[: ids.index(before_id)]
    has_more = len(items) > limit
    page = items[:limit]
    summaries = [PredictionSummary.model_validate(item.model_dump(exclude={"input", "output"})) for item in page]
    return PredictionPage(data=summaries, first_id=page[0].id if page else None, has_more=has_more, last_id=page[-1].id if page else None)


@app.get("/compute/v1/predictions/adme/{id}", response_model=Prediction)
def retrieve(id: str, workspace_id: str | None = None) -> Prediction:
    return get_prediction(id, workspace_id)


@app.post("/compute/v1/predictions/adme/{id}/delete-data", response_model=DeleteResponse)
def delete_data(id: str) -> DeleteResponse:
    item = get_prediction(id)
    if item.data_deleted_at is None:
        item.input = None
        item.output = None
        item.data_deleted_at = now()
    return DeleteResponse(id=item.id, data_deleted_at=item.data_deleted_at)


@app.post("/compute/v1/predictions/adme/estimate-cost", response_model=CostResponse)
def estimate_cost(request: PredictionRequest) -> CostResponse:
    count = len(request.input.molecules)
    total = COST_PER_MOLECULE * count
    return CostResponse(
        breakdown=Breakdown(cost_per_unit_usd=f"{COST_PER_MOLECULE:.4f}", num_units=count),
        estimated_cost_usd=f"{total:.4f}",
    )
