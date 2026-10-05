"""Download the public OpenADMET LogD/permeability model needed by the live test."""

from pathlib import Path
from urllib.request import urlretrieve

REPOSITORY = "openadmet/permeability-logd-ppb-chemeleon-baseline"
FILES = (
    "anvil_training/model.json",
    "anvil_training/model.pth",
    "anvil_training/recipe_components/data.yaml",
    "anvil_training/recipe_components/metadata.yaml",
    "anvil_training/recipe_components/procedure.yaml",
)
DESTINATION = Path("models/permeability-logd-ppb-chemeleon-baseline")


def main():
    for name in FILES:
        destination = DESTINATION / name
        if destination.exists():
            print(f"Using {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/main/{name}"
        print(f"Downloading {url} -> {destination}")
        urlretrieve(url, destination)
    print(f"Model ready at {DESTINATION / 'anvil_training'}")


if __name__ == "__main__":
    main()
