"""Download and verify the public datasets used by the P5 experiments."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive: Path
    sha256: str
    extract_to: Path | None = None


DATASETS = {
    "pva": Dataset(
        name="PVA stress relaxation (Zenodo 21333840)",
        url="https://zenodo.org/api/records/21333840/files/Stress_relaxation_data.xlsx/content",
        archive=EXTERNAL / "pva_gpe_zenodo_21333840" / "Stress_relaxation_data.xlsx",
        sha256="70cfe35e93ee1421fc5ae4c752f61d11fe83972ae36d2d4321d8e511feeb470f",
    ),
    "kupfer": Dataset(
        name="KupferDigital stress relaxation (Zenodo 10820438)",
        url="https://zenodo.org/api/records/10820438/files/Files.zip/content",
        archive=EXTERNAL / "zenodo_10820438" / "Files.zip",
        sha256="4af15b14f0120c58ffee4a2c716a0457450d7819a04720d8723b630b780212f9",
        extract_to=EXTERNAL / "zenodo_10820438" / "extracted",
    ),
    "uci-gas": Dataset(
        name="UCI gas sensor array under flow modulation",
        url="https://archive.ics.uci.edu/static/public/308/gas%2Bsensor%2Barray%2Bunder%2Bflow%2Bmodulation.zip",
        archive=EXTERNAL / "uci_gas_flow_308" / "uci_308.zip",
        sha256="7b062960dbef1a9e8aefc62bc7dbac09bdadb8cbb40793cf2b8e122da1862f90",
        extract_to=EXTERNAL / "uci_gas_flow_308" / "files",
    ),
    "uci-hydraulic": Dataset(
        name="UCI condition monitoring of hydraulic systems",
        url="https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip",
        archive=EXTERNAL / "uci_hydraulic_447" / "uci_447.zip",
        sha256="24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2",
        extract_to=EXTERNAL / "uci_hydraulic_447",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset: Dataset, force: bool) -> None:
    dataset.archive.parent.mkdir(parents=True, exist_ok=True)
    if dataset.archive.exists() and not force:
        if file_sha256(dataset.archive) == dataset.sha256:
            print(f"verified: {dataset.name}")
            return
        raise RuntimeError(
            f"checksum mismatch for {dataset.archive}; rerun with --force to replace it"
        )

    temporary = dataset.archive.with_suffix(dataset.archive.suffix + ".part")
    print(f"downloading: {dataset.name}")
    request = urllib.request.Request(dataset.url, headers={"User-Agent": "DFSC-P5/0.1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if file_sha256(temporary) != dataset.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded checksum mismatch for {dataset.name}")
    temporary.replace(dataset.archive)


def safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if target_root != destination and target_root not in destination.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        bundle.extractall(target)


def prepare(dataset: Dataset, force: bool) -> None:
    download(dataset, force)
    if dataset.extract_to is not None:
        print(f"extracting: {dataset.name}")
        safe_extract(dataset.archive, dataset.extract_to)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="dataset to prepare (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing archive")
    arguments = parser.parse_args()
    selected = DATASETS.values() if arguments.dataset == "all" else (DATASETS[arguments.dataset],)
    for dataset in selected:
        prepare(dataset, arguments.force)


if __name__ == "__main__":
    main()
