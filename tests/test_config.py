from pathlib import Path

import yaml


def test_config_file_exists():
    assert Path("config.yaml").exists(), "El archivo config.yaml debe existir."


def test_config_has_required_sections():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = ["dataset", "split", "model", "mlflow", "artifacts"]

    for section in required_sections:
        assert section in config, f"Falta la sección {section} en config.yaml"


def test_dataset_is_external():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_url = config["dataset"]["url"]

    assert dataset_url.startswith("http"), "El dataset debe venir de una fuente externa."
    assert "sklearn" not in dataset_url.lower(), "No se permite usar sklearn.datasets."
