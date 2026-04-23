from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Utilities
def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _looks_populated(path: str | os.PathLike, min_files: int = 10) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    n = sum(1 for _ in p.rglob("*") if _.is_file())
    return n >= min_files


def _find_raw_file(raw_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = raw_dir / n
        if p.exists():
            return p
    return None


def _raw_dir(cfg: Optional[dict] = None) -> Path:
    if cfg is not None:
        raw = cfg.get("paths", {}).get("raw_root")
        if raw:
            return Path(raw)
    # default: sibling of data/ under the repo root
    here = Path(__file__).resolve().parents[2]
    return here / "raw"


# HAM10000
def download_ham10000(
    dest: str | os.PathLike,
    raw_dir: Optional[str | os.PathLike] = None,
) -> Path:
    dest = _ensure_dir(dest)
    if _looks_populated(dest, min_files=100):
        logger.info("HAM10000 already populated at %s", dest)
        return dest

    raw = Path(raw_dir) if raw_dir else _raw_dir()

    # (a) archive in raw/
    archive = _find_raw_file(raw, [
        "ham10000.zip", "HAM10000.zip",
        "skin-cancer-mnist-ham10000.zip",
    ])
    if archive is not None:
        logger.info("Extracting HAM10000 from %s", archive)
        shutil.unpack_archive(str(archive), str(dest))
        return dest

    # (b) folder in raw/
    folder = None
    for name in ("HAM10000", "ham10000", "skin-cancer-mnist-ham10000"):
        p = raw / name
        if p.is_dir():
            folder = p
            break
    if folder is not None:
        logger.info("Copying HAM10000 from %s", folder)
        for item in folder.iterdir():
            target = dest / item.name
            if not target.exists():
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
        return dest

    # (c) Kaggle
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "HAM10000 missing and kagglehub not installed. Either drop the "
            "Kaggle archive at raw/ham10000.zip or run `pip install kagglehub`."
        ) from exc
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    logger.info("Downloaded HAM10000 to %s (via kagglehub)", path)
    for item in Path(path).iterdir():
        target = dest / item.name
        if not target.exists():
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    return dest


# Derm7pt
def download_derm7pt(
    dest: str | os.PathLike,
    raw_dir: Optional[str | os.PathLike] = None,
    local_archive: Optional[str | os.PathLike] = None,
    gdrive_id: Optional[str] = None,
) -> Path:
    dest = _ensure_dir(dest)
    if (dest / "release_v0" / "meta" / "meta.csv").exists():
        logger.info("Derm7pt already extracted at %s", dest)
        return dest

    if local_archive is None:
        raw = Path(raw_dir) if raw_dir else _raw_dir()
        local_archive = _find_raw_file(raw, [
            "release_v0.zip", "derm7pt.zip", "derm7pt_release_v0.zip",
        ])

    if local_archive is not None:
        archive = Path(local_archive)
        logger.info("Extracting Derm7pt from %s -> %s", archive, dest)
        shutil.unpack_archive(str(archive), str(dest))
        return dest

    if gdrive_id is not None:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError("gdown not installed") from exc
        archive = dest / "derm7pt.zip"
        gdown.download(id=gdrive_id, output=str(archive), quiet=False)
        shutil.unpack_archive(str(archive), str(dest))
        archive.unlink(missing_ok=True)
        return dest

    raise FileNotFoundError(
        "Derm7pt archive not found. Place release_v0.zip at raw/, pass "
        "local_archive=..., or set a gdrive_id. See https://derm.cs.sfu.ca."
    )


# Fitzpatrick17k
FITZ17K_METADATA_URL = (
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/"
    "main/fitzpatrick17k.csv"
)


def download_fitzpatrick17k(
    dest: str | os.PathLike,
    raw_dir: Optional[str | os.PathLike] = None,
    images_url: Optional[str] = None,
    fetch_images_from_urls: bool = False,
    max_images: Optional[int] = None,
    max_workers: int = 8,
) -> Path:
    dest = _ensure_dir(dest)
    raw = Path(raw_dir) if raw_dir else _raw_dir()

    # CSV
    csv_path = dest / "fitzpatrick17k.csv"
    if not csv_path.exists():
        raw_csv = _find_raw_file(raw, ["fitzpatrick17k.csv"])
        if raw_csv is not None:
            shutil.copy2(raw_csv, csv_path)
            logger.info("Copied Fitzpatrick17k metadata from %s", raw_csv)
        else:
            import requests
            logger.info("Downloading Fitzpatrick17k metadata")
            r = requests.get(FITZ17K_METADATA_URL, timeout=60)
            r.raise_for_status()
            csv_path.write_bytes(r.content)

    # Images
    images_dir = dest / "images"
    images_dir.mkdir(exist_ok=True)
    if _looks_populated(images_dir, min_files=100):
        logger.info("Fitzpatrick17k images already present at %s", images_dir)
        return dest

    # (a) archive / folder in raw/
    archive = _find_raw_file(raw, [
        "fitzpatrick17k.zip",
        "fitzpatrick17k_images.zip",
        "fitz17k_images.zip",
    ])
    folder = raw / "fitzpatrick17k_images"
    if archive is not None:
        logger.info("Extracting Fitzpatrick17k images from %s", archive)
        # Groh et al. ship the archive with a nested layout
        # ``data/finalfitz17k/<image>.jpg``. Extract to a scratch dir and
        # then flatten everything into ``images/`` so downstream loaders
        # can find files by basename.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.unpack_archive(str(archive), str(tmp_path))
            candidates = [
                tmp_path / "data" / "finalfitz17k",
                tmp_path / "finalfitz17k",
                tmp_path / "fitzpatrick17k_images",
                tmp_path,
            ]
            src = next(
                (c for c in candidates if c.is_dir() and any(c.iterdir())),
                None,
            )
            if src is None:
                raise RuntimeError(
                    f"No image directory found inside {archive}"
                )
            exts = {".jpg", ".jpeg", ".png"}
            moved = 0
            for img in src.rglob("*"):
                if img.is_file() and img.suffix.lower() in exts:
                    target = images_dir / img.name
                    if target.exists():
                        continue
                    shutil.move(str(img), str(target))
                    moved += 1
            logger.info(
                "Flattened %d Fitzpatrick17k images into %s", moved, images_dir
            )
        return dest
    if folder.is_dir():
        logger.info("Copying Fitzpatrick17k images from %s", folder)
        for item in folder.iterdir():
            if item.is_file():
                shutil.copy2(item, images_dir / item.name)
        return dest

    # (b) external URL
    if images_url is not None:
        import requests
        archive_path = dest / "images.zip"
        try:
            import gdown
            gdown.download(url=images_url, output=str(archive_path),
                           quiet=False, fuzzy=True)
        except ImportError:
            r = requests.get(images_url, stream=True, timeout=600)
            r.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        shutil.unpack_archive(str(archive_path), str(images_dir))
        archive_path.unlink(missing_ok=True)
        return dest

    # (c) per-row URL scrape
    if fetch_images_from_urls:
        _fetch_fitz17k_images_from_csv(
            csv_path, images_dir, max_images=max_images, max_workers=max_workers,
        )
        return dest

    logger.warning(
        "Fitzpatrick17k images not installed. Either place "
        "fitzpatrick17k.zip (or fitzpatrick17k_images.zip) under raw/, "
        "pass images_url=..., or run "
        "download_fitzpatrick17k(..., fetch_images_from_urls=True)."
    )
    return dest


def _fetch_fitz17k_images_from_csv(
    csv_path: Path,
    images_dir: Path,
    max_images: Optional[int] = None,
    max_workers: int = 8,
    timeout: int = 30,
) -> None:
    import pandas as pd
    import requests

    df = pd.read_csv(csv_path)
    if "url" not in df.columns or "md5hash" not in df.columns:
        logger.warning("Fitzpatrick17k CSV missing url / md5hash columns.")
        return
    if max_images is not None:
        df = df.head(max_images)

    existing = {p.name for p in images_dir.iterdir() if p.is_file()}
    todo = [
        (str(row["md5hash"]), str(row["url"]))
        for _, row in df.iterrows()
        if str(row["url"]).startswith("http")
        and f"{row['md5hash']}.jpg" not in existing
    ]
    if not todo:
        logger.info("Fitzpatrick17k: nothing to fetch (all present).")
        return
    logger.info("Fitzpatrick17k: fetching %d images from source URLs...", len(todo))

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (vlm-cbm-derm-fairness)"})

    def _one(pair):
        md5, url = pair
        out = images_dir / f"{md5}.jpg"
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code != 200 or len(r.content) < 1024:
                return (md5, False, f"HTTP {r.status_code}")
            out.write_bytes(r.content)
            return (md5, True, "")
        except Exception as exc:  # pragma: no cover - best-effort scrape
            return (md5, False, str(exc))

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, pair) for pair in todo]
        for fut in as_completed(futures):
            _, success, _err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
    logger.info("Fitzpatrick17k fetch: %d ok / %d failed.", ok, fail)


# DDI (Diverse Dermatology Images)
def download_ddi(
    dest: str | os.PathLike,
    raw_dir: Optional[str | os.PathLike] = None,
    images_url: Optional[str] = None,
) -> Path:
    dest = _ensure_dir(dest)
    if (dest / "ddi_metadata.csv").exists() and _looks_populated(dest, min_files=100):
        logger.info("DDI already populated at %s", dest)
        return dest

    if raw_dir is None:
        raw_dir = _raw_dir()
    raw = Path(raw_dir)

    archive = _find_raw_file(raw, [
        "ddidiversedermatologyimages.zip",
        "ddi.zip", "DDI.zip", "ddi_images.zip",
    ])
    if archive is not None:
        logger.info("Extracting DDI from %s -> %s", archive, dest)
        shutil.unpack_archive(str(archive), str(dest))
        return dest

    if images_url is not None:
        import requests
        archive = dest / "ddi_archive.zip"
        try:
            import gdown
            gdown.download(url=images_url, output=str(archive),
                           quiet=False, fuzzy=True)
        except ImportError:
            r = requests.get(images_url, stream=True, timeout=600)
            r.raise_for_status()
            with open(archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        shutil.unpack_archive(str(archive), str(dest))
        archive.unlink(missing_ok=True)
        return dest

    raise FileNotFoundError(
        "DDI archive not found. Drop the Stanford AIMI ZIP at "
        "raw/ddidiversedermatologyimages.zip or pass images_url=..."
    )


# Driver
def download_all(cfg: dict, raw_dir: Optional[str | os.PathLike] = None) -> None:
    paths = cfg["paths"]["datasets"]
    raw_dir = raw_dir or cfg.get("paths", {}).get("raw_root") or _raw_dir(cfg)

    try:
        download_ham10000(paths["ham10000"], raw_dir=raw_dir)
    except Exception as exc:
        logger.warning("Skipping HAM10000: %s", exc)

    try:
        download_derm7pt(paths["derm7pt"], raw_dir=raw_dir)
    except FileNotFoundError as exc:
        logger.warning("Skipping Derm7pt: %s", exc)

    try:
        download_fitzpatrick17k(paths["fitzpatrick17k"], raw_dir=raw_dir)
    except Exception as exc:
        logger.warning("Skipping Fitzpatrick17k: %s", exc)

    try:
        download_ddi(paths["ddi"], raw_dir=raw_dir)
    except FileNotFoundError as exc:
        logger.warning("Skipping DDI: %s", exc)


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    download_all(cfg)
