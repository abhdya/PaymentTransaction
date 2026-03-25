from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_data import generate_and_write
from .reconcile import load_bank_csv, load_platform_csv, reconcile_month, write_outputs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recon", description="Month-end reconciliation demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate-data", help="Generate synthetic CSV data")
    gen.add_argument("--month", required=True, help="YYYY-MM")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--data-dir", default="data", help="output directory for CSVs")
    gen.add_argument("--n-customers", type=int, default=50)
    gen.add_argument("--n-charges", type=int, default=300)
    gen.add_argument("--duplicate-in", choices=["bank", "platform"], default="bank")

    rec = sub.add_parser("reconcile", help="Run reconciliation and write reports")
    rec.add_argument("--month", required=True, help="YYYY-MM")
    rec.add_argument("--data-dir", default="data", help="input directory containing CSVs")
    rec.add_argument("--out-dir", default="out", help="output directory for reports")

    demo = sub.add_parser("demo", help="Generate data then reconcile")
    demo.add_argument("--month", required=True, help="YYYY-MM")
    demo.add_argument("--seed", type=int, required=True)
    demo.add_argument("--data-dir", default="data")
    demo.add_argument("--out-dir", default="out")
    demo.add_argument("--n-customers", type=int, default=50)
    demo.add_argument("--n-charges", type=int, default=300)
    demo.add_argument("--duplicate-in", choices=["bank", "platform"], default="bank")

    return p


def _print_summary(out: dict) -> None:
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "generate-data":
        out = generate_and_write(
            month=args.month,
            seed=args.seed,
            data_dir=Path(args.data_dir),
            n_customers=args.n_customers,
            n_charges=args.n_charges,
            duplicate_in=args.duplicate_in,
        )
        _print_summary(out)
        return 0

    if args.cmd in {"reconcile", "demo"}:
        if args.cmd == "demo":
            gen_out = generate_and_write(
                month=args.month,
                seed=args.seed,
                data_dir=Path(args.data_dir),
                n_customers=args.n_customers,
                n_charges=args.n_charges,
                duplicate_in=args.duplicate_in,
            )
        else:
            gen_out = None

        data_dir = Path(args.data_dir)
        platform_path = data_dir / "platform_transactions.csv"
        bank_path = data_dir / "bank_settlements.csv"
        platform_df = load_platform_csv(platform_path)
        bank_df = load_bank_csv(bank_path)
        result = reconcile_month(platform_df=platform_df, bank_df=bank_df, month=args.month)
        paths = write_outputs(result, out_dir=Path(args.out_dir))

        out = {"outputs": paths, "summary": result.summary}
        if gen_out is not None:
            out["generated"] = gen_out
        _print_summary(out)
        return 0

    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

