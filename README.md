# OpenADMET Experimental Boltz-Compatible REST API

An experimental FastAPI adapter exposing the [Boltz ADME endpoint shape](https://api.boltz.com/docs/api/api/resources/predictions/subresources/adme/) over trained
[OpenADMET Models](https://github.com/OpenADMET/openadmet-models) Anvil models.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with a running Linux-container
  engine (Docker Engine on Linux or Docker Desktop on macOS)
- [Pixi](https://pixi.sh)
- At least 20 GB free for the upstream OpenADMET image and build cache

The Pixi environment is configured in `pyproject.toml` and supports macOS
(Intel/Apple Silicon) and Linux (x86-64/ARM64).

The upstream image is amd64-only, so the commands below request `linux/amd64`.
ARM hosts need amd64 emulation (included in Docker Desktop; on ARM Linux,
configure QEMU/binfmt support for Docker). Emulated inference may be slower.

## Start the API container

From this repository, initialize the Python environment and download OpenADMET's
public LogD/Caco-2 permeability model (about 42 MB):

```bash
pixi install
pixi run download-model
docker info
```

Build the API image. The first build downloads the approximately 8 GB OpenADMET
base image and can take several minutes; subsequent builds use the cache.

```bash
docker build --platform linux/amd64 --tag adme-api:local .
```

Start a named container, mounting the downloaded model read-only:

```bash
MODEL_DIR="$PWD/models/permeability-logd-ppb-chemeleon-baseline/anvil_training"

docker run --detach \
  --platform linux/amd64 \
  --name adme-api \
  --publish 127.0.0.1:8000:8000 \
  --mount "type=bind,source=$MODEL_DIR,target=/models/adme,readonly" \
  --env ADME_MODEL_DIRS=/models/adme \
  --env ADME_LIPOPHILICITY_COLUMN=OADMET_PRED_chemprop_logD \
  --env ADME_PERMEABILITY_COLUMN=OADMET_PRED_chemprop_caco2_atob_LogPapp \
  --env ADME_SOLUBILITY_COLUMN=OADMET_PRED_chemprop_logD \
  --env ADME_SOLUBILITY_MODE=logd \
  --env ADME_MODEL_VERSION=permeability-logd-ppb-chemeleon-baseline \
  --env ADME_ACCELERATOR=cpu \
  adme-api:local
```

Check that it is running:

```bash
docker logs adme-api
```

Open <http://127.0.0.1:8000/docs> in your browser.

Prediction records are held in memory and are cleared when the container stops.

## Run the Boltz SDK inference script

[`tests/live_test.py`](tests/live_test.py) uses `boltz-api` with the local base URL:

```python
client = Boltz(
    api_key="local-test",
    base_url="http://127.0.0.1:8000",
)
```

Run it against the container:

```bash
pixi run test-live
```

The script predicts LogD and Caco-2 A-to-B permeability for caffeine, aspirin,
and ibuprofen. It also verifies typed SDK parsing, response order, finite values,
idempotency, retrieval, listing, cost estimation, and data deletion. Example output:

```text
Boltz SDK live ADME test passed against http://127.0.0.1:8000
  caffeine  logD=-0.7718 permeability=-5.876 solubility=high-confidence
  aspirin   logD=-2.965 permeability=-6.18 solubility=high-confidence
  ibuprofen logD=1.035 permeability=-5.059 solubility=medium-confidence
```

To target another deployment:

```bash
ADME_API_URL=http://example.test:8000 pixi run test-live
```

## Stop and remove the container

```bash
docker stop adme-api
docker rm adme-api
```

The image remains cached for the next run. Remove it separately only when you want
to reclaim its disk space.

## One-command smoke test

To download the model, build a temporary container, run the SDK inference test,
and remove the container automatically:

```bash
pixi run test-live-container
```

## Model mapping

The public
[`permeability-logd-ppb-chemeleon-baseline`](https://huggingface.co/openadmet/permeability-logd-ppb-chemeleon-baseline)
model provides LogD and Caco-2 permeability predictions. The API derives its
solubility judgement from LogD: high confidence at <=1, medium confidence between
1 and 3, and high risk at >=3. This is a risk heuristic, not a separately trained
solubility model.

For other Anvil models, configure `ADME_MODEL_DIRS` and the three
`ADME_*_COLUMN` variables with their generated prediction columns.

## Unit tests

Unit and API contract tests do not require model weights or a container:

```bash
pixi run test
```
