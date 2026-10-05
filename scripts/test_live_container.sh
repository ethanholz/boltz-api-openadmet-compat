#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
name=adme-api-live-test
image=adme-api:local
model="$PWD/models/permeability-logd-ppb-chemeleon-baseline/anvil_training"

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ "$status" -ne 0 ]; then
        docker logs "$name" 2>/dev/null || true
    fi
    docker rm --force "$name" >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT INT TERM

python scripts/download_model.py
docker rm --force "$name" >/dev/null 2>&1 || true
docker build --platform linux/amd64 --tag "$image" .
docker run --detach --platform linux/amd64 --name "$name" --publish 127.0.0.1:8000:8000 \
    --mount "type=bind,source=$model,target=/models/adme,readonly" \
    --env ADME_MODEL_DIRS=/models/adme \
    --env ADME_LIPOPHILICITY_COLUMN=OADMET_PRED_chemprop_logD \
    --env ADME_PERMEABILITY_COLUMN=OADMET_PRED_chemprop_caco2_atob_LogPapp \
    --env ADME_SOLUBILITY_COLUMN=OADMET_PRED_chemprop_logD \
    --env ADME_SOLUBILITY_MODE=logd \
    --env ADME_MODEL_VERSION=permeability-logd-ppb-chemeleon-baseline \
    --env ADME_ACCELERATOR=cpu \
    "$image"

for _ in $(seq 1 60); do
    if python -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8000/openapi.json", timeout=1)' 2>/dev/null; then
        python tests/live_test.py
        exit
    fi
    sleep 1
done

echo "API did not become ready within 60 seconds" >&2
exit 1
