"""Command-line entry point for the LangPro dataset expansion pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent


def run_stage(command: Sequence[str], description: str) -> None:
    """Run one pipeline stage and stop if it fails."""
    print(f"\n[{description}] {' '.join(command)}")
    started = time.time()
    subprocess.run(list(command), check=True)
    print(f"[{description}] completed in {(time.time() - started) / 60:.1f} min")


def export_alphafold_ids(input_csv: Path, output_file: Path) -> None:
    """Write the unique UniProt identifiers required for AlphaFold retrieval."""
    frame = pd.read_csv(input_csv, usecols=["uniprot_id"])
    identifiers = sorted(
        {
            str(value).strip()
            for value in frame["uniprot_id"].dropna()
            if str(value).strip()
        }
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    print(f"Wrote {len(identifiers)} UniProt identifiers to {output_file}")


def check_alphafold_entry(uniprot_id: str) -> None:
    """Report the current AlphaFold API entry and downloadable structure URLs."""
    accession = uniprot_id.strip().upper()
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"AlphaFold API returned HTTP {exc.code} for {accession}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach the AlphaFold API: {exc.reason}") from exc

    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise SystemExit(f"No AlphaFold entry was found for {accession}")

    for record in records:
        print(f"UniProt: {record.get('uniprotAccession', accession)}")
        for field in ("pdbUrl", "cifUrl", "bcifUrl", "paeDocUrl"):
            if record.get(field):
                print(f"{field}: {record[field]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add chemical, conservation, and structural features to a LangPro dataset."
    )
    parser.add_argument("--input-csv", type=Path, help="Input dataset CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset"),
        help="Directory for stage outputs (default: outputs/dataset).",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=None,
        help="Pipeline stages to run in order (default: 1 2).",
    )
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--esm-cache-dir", type=Path)
    parser.add_argument(
        "--af-dir",
        type=Path,
        default=Path("outputs/alphafold"),
        help="AlphaFold structure cache (default: outputs/alphafold).",
    )
    parser.add_argument("--download-af", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-dssp", action="store_true")
    parser.add_argument(
        "--export-af-ids",
        type=Path,
        metavar="PATH",
        help="Export unique UniProt identifiers from the input CSV.",
    )
    parser.add_argument(
        "--check-alphafold",
        metavar="UNIPROT_ID",
        help="Query one UniProt identifier through the public AlphaFold API.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.check_alphafold:
        check_alphafold_entry(args.check_alphafold)
        if args.input_csv is None:
            return

    if args.input_csv is None:
        raise SystemExit("--input-csv is required to run stages or export identifiers.")
    input_csv = args.input_csv.resolve()
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    if args.export_af_ids:
        export_alphafold_ids(input_csv, args.export_af_ids)
        if args.phases is None:
            return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    af_dir = args.af_dir.resolve()
    python = sys.executable

    phase1_output = output_dir / "dataset_phase1.csv"
    phase2_output = output_dir / "dataset_expanded.csv"
    phase3_output = output_dir / "dataset_phase3.csv"
    current_input = input_csv

    phases = args.phases or (1, 2)
    for phase in phases:
        if phase == 1:
            command = [
                python,
                "-m",
                "dataset.expansion.phase1",
                "--input",
                str(current_input),
                "--output",
                str(phase1_output),
            ]
            run_stage(command, "phase 1: chemical descriptors")
            current_input = phase1_output

        elif phase == 2:
            command = [
                python,
                "-m",
                "dataset.expansion.phase2",
                "--input",
                str(current_input),
                "--output",
                str(phase2_output),
                "--esm_model",
                args.esm_model,
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
            ]
            if args.esm_cache_dir:
                command.extend(["--esm_cache_dir", str(args.esm_cache_dir.resolve())])
            run_stage(command, "phase 2: conservation scores")
            current_input = phase2_output

        else:
            command = [
                python,
                "-m",
                "dataset.expansion.phase3",
                "--input_csv",
                str(current_input),
                "--output_csv",
                str(phase3_output),
                "--af_dir",
                str(af_dir),
                "--compute_features",
                "--num_workers",
                str(args.workers),
            ]
            if args.download_af:
                command.append("--download_af")
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            if args.no_dssp:
                command.append("--no_dssp")
            run_stage(command, "phase 3: structural features")
            current_input = phase3_output

    print(f"Final dataset: {current_input}")


if __name__ == "__main__":
    main()
