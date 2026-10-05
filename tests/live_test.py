"""End-to-end inference through the published Boltz Python SDK."""

import math
import os
import time
from uuid import uuid4

from boltz_api import Boltz

BASE_URL = os.getenv("ADME_API_URL", "http://127.0.0.1:8000")
TIMEOUT = float(os.getenv("ADME_TEST_TIMEOUT", "120"))
MOLECULES = [
    {"id": "caffeine", "smiles": "CN1C(=O)N(C)c2ncn(C)c2C1=O"},
    {"id": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"id": "ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"},
]


def main():
    with Boltz(
        api_key=os.getenv("BOLTZ_API_KEY", "local-test"),
        base_url=BASE_URL,
        timeout=TIMEOUT,
    ) as client:
        key = f"live-{uuid4()}"
        request = {"input": {"molecules": MOLECULES}, "model": "adme-v1"}

        estimate = client.predictions.adme.estimate_cost(**request)
        assert estimate.breakdown.application == "adme"
        assert estimate.breakdown.num_units == len(MOLECULES)
        assert isinstance(estimate.estimated_cost_usd, str)
        float(estimate.estimated_cost_usd)

        prediction = client.predictions.adme.start(**request, idempotency_key=key)
        deadline = time.monotonic() + TIMEOUT
        while prediction.status in {"pending", "running"} and time.monotonic() < deadline:
            time.sleep(0.5)
            prediction = client.predictions.adme.retrieve(prediction.id)

        assert prediction.status == "succeeded", prediction.error
        assert prediction.output is not None
        outputs = prediction.output.molecules
        assert [item.external_id for item in outputs] == [item["id"] for item in MOLECULES]
        assert [item.smiles for item in outputs] == [item["smiles"] for item in MOLECULES]
        for item in outputs:
            assert item.status == "succeeded", item.error
            assert item.error is None
            assert item.adme is not None
            assert item.adme.solubility in {"high-confidence", "medium-confidence", "high-risk"}
            assert math.isfinite(item.adme.lipophilicity)
            assert math.isfinite(item.adme.permeability)

        repeated = client.predictions.adme.start(**request, idempotency_key=key)
        assert repeated.id == prediction.id
        retrieved = client.predictions.adme.retrieve(prediction.id)
        assert retrieved.output is not None
        assert retrieved.output.to_dict() == prediction.output.to_dict()

        page = client.predictions.adme.list(limit=100)
        assert prediction.id in {item.id for item in page.data}

        deleted = client.predictions.adme.delete_data(prediction.id)
        assert deleted.data_deleted is True
        retained = client.predictions.adme.retrieve(prediction.id)
        assert retained.input is None and retained.output is None

        print(f"Boltz SDK live ADME test passed against {BASE_URL}")
        for item in outputs:
            adme = item.adme
            print(
                f"  {item.external_id:9} logD={adme.lipophilicity:.4g} "
                f"permeability={adme.permeability:.4g} solubility={adme.solubility}"
            )


if __name__ == "__main__":
    main()
