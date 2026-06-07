"""Download datasets from Kaggle and Hugging Face into isolated temp directories."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd


KAGGLE_URL_RE = re.compile(
    r'(?:https?://)?(?:www\.)?kaggle\.com/datasets/(?P<owner>[^/]+)/(?P<name>[^/?#]+)',
    re.IGNORECASE,
)
HF_URL_RE = re.compile(
    r'(?:https?://)?(?:huggingface\.co/)?datasets/(?P<ns>[^/]+)/(?P<name>[^/?#]+)',
    re.IGNORECASE,
)


def parse_kaggle_ref(ref: str) -> tuple[str, str]:
    """Return (owner/dataset, display_name) from name or URL."""
    ref = (ref or '').strip()
    if not ref:
        raise ValueError('Kaggle dataset reference is empty')
    match = KAGGLE_URL_RE.search(ref)
    if match:
        slug = f"{match.group('owner')}/{match.group('name')}"
        return slug, slug
    if re.match(r'^[\w-]+/[\w-]+$', ref):
        return ref, ref
    raise ValueError(
        'Invalid Kaggle reference. Use owner/dataset or a Kaggle datasets URL.'
    )


def parse_hf_ref(ref: str) -> tuple[str, str]:
    """Return (namespace/dataset, display_name) from id or URL."""
    ref = (ref or '').strip()
    if not ref:
        raise ValueError('Hugging Face dataset reference is empty')
    match = HF_URL_RE.search(ref)
    if match:
        slug = f"{match.group('ns')}/{match.group('name')}"
        return slug, slug
    if re.match(r'^[\w.-]+/[\w.-]+$', ref):
        return ref, ref
    raise ValueError(
        'Invalid Hugging Face reference. Use namespace/dataset or a HF datasets URL.'
    )


def _find_csv_file(directory: Path) -> Path:
    csv_files = sorted(directory.rglob('*.csv'))
    if not csv_files:
        raise ValueError('No CSV file found in downloaded dataset')
    return csv_files[0]


def _make_temp_dir(prefix: str) -> Path:
    base = Path(tempfile.mkdtemp(prefix=f'cml_{prefix}_'))
    return base


def download_kaggle_dataset(ref: str, dest_dir: Path | None = None) -> tuple[Path, str, Path]:
    """
    Download Kaggle dataset to dest_dir (or new temp dir).
    Returns (csv_path, display_name, temp_root).
    """
    slug, display = parse_kaggle_ref(ref)
    temp_root = dest_dir or _make_temp_dir('kaggle')
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            'kagglehub is not installed. Add it to requirements and pip install.'
        ) from exc

    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_json.exists():
            raise ValueError(
                'Kaggle credentials required. Set KAGGLE_USERNAME and KAGGLE_KEY '
                'environment variables or place kaggle.json in ~/.kaggle/'
            )

    download_path = Path(kagglehub.dataset_download(slug))
    if not download_path.exists():
        raise ValueError(f'Kaggle download failed for {slug}')

    work_dir = temp_root / 'data'
    if download_path.is_dir():
        shutil.copytree(download_path, work_dir, dirs_exist_ok=True)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download_path, work_dir / download_path.name)

    csv_path = _find_csv_file(work_dir)
    return csv_path, display, temp_root


def _dataset_to_csv_chunks(dataset, csv_path: Path, max_rows: int = 100_000) -> int:
    """Stream Hugging Face dataset split to CSV; return row count."""
    first = True
    rows = 0
    for batch in dataset.iter(batch_size=5000):
        df = pd.DataFrame(batch)
        if first:
            df.to_csv(csv_path, index=False, mode='w')
            first = False
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False)
        rows += len(df)
        if rows >= max_rows:
            break
    return rows


def download_hf_dataset(
    ref: str,
    dest_dir: Path | None = None,
    max_rows: int = 100_000,
    streaming: bool = True,
) -> tuple[Path, str, Path]:
    """
    Load Hugging Face dataset (streaming for large sets), save as CSV.
    Returns (csv_path, display_name, temp_root).
    """
    slug, display = parse_hf_ref(ref)
    temp_root = dest_dir or _make_temp_dir('hf')
    temp_root.mkdir(parents=True, exist_ok=True)
    csv_path = temp_root / f'{slug.replace("/", "_")}.csv'

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            'datasets library is not installed. Add it to requirements and pip install.'
        ) from exc

    try:
        ds = load_dataset(slug, split='train', streaming=streaming)
    except Exception:
        ds = load_dataset(slug, split='train', streaming=False)

    if hasattr(ds, 'iter'):
        _dataset_to_csv_chunks(ds, csv_path, max_rows=max_rows)
    else:
        df = ds.to_pandas()
        if len(df) > max_rows:
            df = df.head(max_rows)
        df.to_csv(csv_path, index=False)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise ValueError(f'Failed to export Hugging Face dataset {slug} to CSV')

    return csv_path, display, temp_root


def cleanup_temp_dir(temp_root: Path | None) -> None:
    if temp_root and temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
