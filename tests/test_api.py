import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

from fastapi.testclient import TestClient

from app import Error, OpenAdmetPredictor, PredictionRequest, Store, app, start


def predictor(smiles):
    return [
        Error(code="invalid_smiles", message="SMILES could not be featurized")
        if value == "invalid"
        else (1.25, 2.5, "medium-confidence")
        for value in smiles
    ]


def client():
    app.state.store = Store()
    app.state.predictor = predictor
    return TestClient(app)


def request(molecules=None, **extra):
    return {
        "input": {"molecules": molecules if molecules is not None else [{"smiles": "CCO", "id": "client-1"}]},
        "model": "adme-v1",
        **extra,
    }


def test_prediction_lifecycle_and_idempotency():
    with client() as api:
        response = api.post("/compute/v1/predictions/adme", json=request(idempotency_key="retry-1"))
        assert response.status_code == 200
        prediction = response.json()
        assert prediction["status"] == "succeeded"
        assert prediction["output"]["molecules"][0] == {
            "id": prediction["output"]["molecules"][0]["id"],
            "adme": {"lipophilicity": 1.25, "permeability": 2.5, "solubility": "medium-confidence"},
            "error": None,
            "smiles": "CCO",
            "status": "succeeded",
            "external_id": "client-1",
        }

        repeated = api.post("/compute/v1/predictions/adme", json=request(idempotency_key="retry-1"))
        assert repeated.json()["id"] == prediction["id"]

        retrieved = api.get(f"/compute/v1/predictions/adme/{prediction['id']}")
        assert retrieved.json() == prediction

        page = api.get("/compute/v1/predictions/adme").json()
        assert [item["id"] for item in page["data"]] == [prediction["id"]]
        assert "input" not in page["data"][0]
        assert page["first_id"] == page["last_id"] == prediction["id"]
        assert page["has_more"] is False

        deleted = api.post(f"/compute/v1/predictions/adme/{prediction['id']}/delete-data").json()
        assert deleted["data_deleted"] is True
        after_delete = api.get(f"/compute/v1/predictions/adme/{prediction['id']}").json()
        assert after_delete["input"] is None
        assert after_delete["output"] is None
        assert after_delete["data_deleted_at"] == deleted["data_deleted_at"]


def test_concurrent_idempotent_retries_create_once():
    app.state.store = Store()
    calls = []

    def slow_predictor(smiles):
        calls.append(smiles)
        time.sleep(0.02)
        return [(1.0, 2.0, "high-confidence")]

    app.state.predictor = slow_predictor
    payload = PredictionRequest.model_validate(request(idempotency_key="simultaneous"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: start(payload), range(8)))
    assert len({result.id for result in results}) == 1
    assert len(app.state.store.items) == 1
    assert len(calls) == 1


def test_per_molecule_failure_preserves_order():
    with client() as api:
        molecules = [{"smiles": "invalid", "id": "bad"}, {"smiles": "CC"}]
        result = api.post("/compute/v1/predictions/adme", json=request(molecules)).json()
        outputs = result["output"]["molecules"]
        assert [item["smiles"] for item in outputs] == ["invalid", "CC"]
        assert outputs[0]["status"] == "failed"
        assert outputs[0]["adme"] is None
        assert outputs[0]["error"]["code"] == "invalid_smiles"
        assert outputs[1]["status"] == "succeeded"


def test_cost_and_request_validation():
    with client() as api:
        estimate = api.post(
            "/compute/v1/predictions/adme/estimate-cost",
            json=request([{"smiles": "C"}, {"smiles": "CC"}]),
        )
        assert estimate.status_code == 200
        assert estimate.json()["breakdown"] == {
            "application": "adme", "cost_per_unit_usd": "0.0500", "num_units": 2
        }
        assert estimate.json()["estimated_cost_usd"] == "0.1000"

        assert api.post("/compute/v1/predictions/adme", json=request([])).status_code == 422
        too_many = [{"smiles": "C"}] * 129
        assert api.post("/compute/v1/predictions/adme", json=request(too_many)).status_code == 422
        assert api.post("/compute/v1/predictions/adme", json={**request(), "model": "other"}).status_code == 422


def test_openadmet_adapter(monkeypatch):
    monkeypatch.setenv("ADME_MODEL_DIRS", "/models/one:/models/two")
    monkeypatch.setenv("ADME_LIPOPHILICITY_COLUMN", "logd")
    monkeypatch.setenv("ADME_PERMEABILITY_COLUMN", "perm")
    monkeypatch.setenv("ADME_SOLUBILITY_COLUMN", "sol")
    monkeypatch.setenv("ADME_SOLUBILITY_MODE", "logd")
    calls = {}

    class Frame:
        at = {
            (0, "logd"): 1.0, (0, "perm"): 2.0, (0, "sol"): 0.8,
            (1, "logd"): None, (1, "perm"): None, (1, "sol"): None,
            (2, "logd"): 3.0, (2, "perm"): 4.0, (2, "sol"): 3.1,
        }

    pandas = ModuleType("pandas")
    pandas.DataFrame = lambda data: data
    pandas.isna = lambda value: value is None
    inference = ModuleType("openadmet.models.inference.inference")

    def fake_predict(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return Frame()

    inference.predict = fake_predict
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.setitem(sys.modules, "openadmet", ModuleType("openadmet"))
    monkeypatch.setitem(sys.modules, "openadmet.models", ModuleType("openadmet.models"))
    monkeypatch.setitem(sys.modules, "openadmet.models.inference", ModuleType("openadmet.models.inference"))
    monkeypatch.setitem(sys.modules, "openadmet.models.inference.inference", inference)

    results = OpenAdmetPredictor()(["C", "invalid", "CC"])
    assert results[0] == (1.0, 2.0, "high-confidence")
    assert isinstance(results[1], Error) and results[1].code == "invalid_smiles"
    assert results[2] == (3.0, 4.0, "high-risk")
    assert calls["args"][1:] == ("smiles", ["/models/one", "/models/two"])
    assert calls["kwargs"] == {"accelerator": "cpu", "log": False}


def test_list_pagination_and_workspace_filter():
    with client() as api:
        first = api.post("/compute/v1/predictions/adme", json=request(workspace_id="one")).json()
        second = api.post("/compute/v1/predictions/adme", json=request(workspace_id="one")).json()
        api.post("/compute/v1/predictions/adme", json=request(workspace_id="two"))

        page = api.get("/compute/v1/predictions/adme", params={"workspace_id": "one", "limit": 1}).json()
        assert page["data"][0]["id"] == first["id"]
        assert page["has_more"] is True
        next_page = api.get("/compute/v1/predictions/adme", params={"workspace_id": "one", "after_id": first["id"]}).json()
        assert [item["id"] for item in next_page["data"]] == [second["id"]]
