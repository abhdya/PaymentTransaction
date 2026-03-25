from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

UTC = timezone.utc


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(UTC)


def _month_str(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _ensure_int_series(s: pd.Series) -> pd.Series:
    # Keep NaNs as NaN; otherwise coerce to int.
    if s.isna().any():
        return pd.to_numeric(s, errors="raise")
    return pd.to_numeric(s, errors="raise").astype(int)


Status = Literal[
    "matched",
    "late_settlement",
    "missing_settlement",
    "bank_duplicate",
    "platform_duplicate",
    "refund_without_original",
    "amount_mismatch",
    "currency_mismatch",
    "rounding_batch_diff",
    "unmatched_bank_entry",
]


@dataclass(frozen=True)
class ReconResult:
    report_df: pd.DataFrame
    summary: dict[str, Any]


def load_platform_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tx_id", "created_at", "type", "currency", "amount_cents", "customer_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"platform missing required columns: {sorted(missing)}")
    if "original_tx_id" not in df.columns:
        df["original_tx_id"] = ""

    df["tx_id"] = df["tx_id"].astype(str)
    df["created_at_dt"] = df["created_at"].map(_dt)
    df["amount_cents"] = _ensure_int_series(df["amount_cents"])
    df["currency"] = df["currency"].astype(str)
    df["type"] = df["type"].astype(str)
    df["original_tx_id"] = df["original_tx_id"].fillna("").astype(str)
    df["created_month"] = df["created_at_dt"].map(_month_str)
    return df


def load_bank_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"bank_id", "settled_at", "currency", "amount_cents", "tx_id", "batch_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"bank missing required columns: {sorted(missing)}")
    if "entry_type" not in df.columns:
        df["entry_type"] = "per_tx"

    df["bank_id"] = df["bank_id"].astype(str)
    df["settled_at_dt"] = df["settled_at"].map(_dt)
    df["amount_cents"] = _ensure_int_series(df["amount_cents"])
    df["currency"] = df["currency"].astype(str)
    df["tx_id"] = df["tx_id"].fillna("").astype(str)
    df["batch_id"] = df["batch_id"].astype(str)
    df["entry_type"] = df["entry_type"].fillna("per_tx").astype(str)
    df["settled_month"] = df["settled_at_dt"].map(_month_str)
    return df


def reconcile_month(
    *,
    platform_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    month: str,
    rounding_threshold_cents: int = 5,
) -> ReconResult:
    """
    Reconcile a target month:
    - Platform transactions created in `month` must have a settlement (possibly in later month).
    - Bank entries in `month` should map to platform transactions or batches.
    """
    plat_m = platform_df[platform_df["created_month"] == month].copy()
    bank_any = bank_df.copy()

    # Duplicate detection
    plat_dup_tx_ids = set(
        plat_m["tx_id"][plat_m["tx_id"].duplicated(keep=False)].astype(str).tolist()
    )
    bank_dup_bank_ids = set(
        bank_any["bank_id"][bank_any["bank_id"].duplicated(keep=False)].astype(str).tolist()
    )
    # Also detect bank duplicates by tx_id for per_tx entries (more meaningful for reconciliation).
    bank_per_tx = bank_any[(bank_any["entry_type"] == "per_tx") & (bank_any["tx_id"] != "")]
    bank_dup_tx_ids = set(
        bank_per_tx["tx_id"][bank_per_tx["tx_id"].duplicated(keep=False)].astype(str).tolist()
    )

    # Match per-tx settlements by tx_id
    bank_per_tx_idx = bank_per_tx.set_index("tx_id", drop=False)
    bank_tx_id_set = set(bank_per_tx_idx.index.astype(str).tolist())

    # If the bank provides batch_total entries in the target month, assume platform charges that are
    # missing per-tx settlements are covered by those batch totals (synthetic-data-friendly).
    bank_batch_in_month = bank_any[
        (bank_any["entry_type"] == "batch_total") & (bank_any["settled_month"] == month)
    ].copy()
    inferred_batch_covered_tx_ids: set[str] = set()
    if not bank_batch_in_month.empty:
        inferred_batch_covered_tx_ids = set(
            plat_m[
                (plat_m["type"] == "charge")
                & (~plat_m["tx_id"].astype(str).isin(bank_tx_id_set))
            ]["tx_id"]
            .astype(str)
            .tolist()
        )

    # Precompute charge ids for refund validation
    all_charge_ids = set(platform_df[platform_df["type"] == "charge"]["tx_id"].astype(str).tolist())

    report_rows: list[dict[str, Any]] = []

    for _, tx in plat_m.iterrows():
        this_tx_id = str(tx["tx_id"])
        status: Status
        reason: str
        matched_bank_id = ""
        settled_at = ""
        settled_month = ""

        if this_tx_id in plat_dup_tx_ids:
            status = "platform_duplicate"
            reason = "duplicate tx_id in platform month subset"
        else:
            if this_tx_id in bank_tx_id_set:
                # There may be multiple bank rows if duplicate in bank; handle explicitly
                bank_matches = bank_per_tx[bank_per_tx["tx_id"] == this_tx_id].sort_values(
                    "settled_at_dt"
                )
                if len(bank_matches) > 1 or this_tx_id in bank_dup_tx_ids:
                    status = "bank_duplicate"
                    reason = "multiple bank per_tx settlements share tx_id"
                    matched_bank_id = ",".join(bank_matches["bank_id"].astype(str).tolist())
                    # still capture earliest settle month for reporting
                    settled_at = str(bank_matches.iloc[0]["settled_at"])
                    settled_month = str(bank_matches.iloc[0]["settled_month"])
                else:
                    bank_row = bank_matches.iloc[0]
                    matched_bank_id = str(bank_row["bank_id"])
                    settled_at = str(bank_row["settled_at"])
                    settled_month = str(bank_row["settled_month"])
                    # Currency mismatch takes precedence over timing status.
                    if str(bank_row["currency"]) != str(tx["currency"]):
                        status = "currency_mismatch"
                        reason = (
                            f"currency mismatch: platform={tx['currency']} "
                            f"bank={bank_row['currency']}"
                        )
                    # Amount mismatch: bank settled a different value for the same tx_id.
                    elif int(bank_row["amount_cents"]) != int(tx["amount_cents"]):
                        status = "amount_mismatch"
                        reason = (
                            f"amount mismatch: platform={tx['amount_cents']} cents "
                            f"bank={bank_row['amount_cents']} cents "
                            f"delta={int(bank_row['amount_cents']) - int(tx['amount_cents'])} cents"
                        )
                    elif settled_month != month:
                        status = "late_settlement"
                        reason = "settled after month end"
                    else:
                        status = "matched"
                        reason = "matched by tx_id"
            else:
                status = "missing_settlement"
                reason = "no bank per_tx entry with this tx_id"

        if status == "missing_settlement" and this_tx_id in inferred_batch_covered_tx_ids:
            status = "matched"
            reason = "covered by bank batch_total entry"

        # Refund validation: orthogonal to timing/duplicate checks.
        # If a higher-priority issue was already flagged (bank_duplicate, platform_duplicate,
        # amount_mismatch, currency_mismatch) we keep that status and append the refund issue to
        # the reason rather than silently overriding it.
        if str(tx["type"]) == "refund":
            orig = str(tx.get("original_tx_id", "") or "")
            if orig and orig not in all_charge_ids:
                if status in {"bank_duplicate", "platform_duplicate", "amount_mismatch", "currency_mismatch"}:
                    reason = reason + f"; also: refund original_tx_id not found: {orig}"
                else:
                    status = "refund_without_original"
                    reason = f"refund original_tx_id not found: {orig}"

        report_rows.append(
            {
                "entity": "platform_tx",
                "month": month,
                "tx_id": this_tx_id,
                "bank_id": matched_bank_id,
                "batch_id": "",
                "created_at": str(tx["created_at"]),
                "settled_at": settled_at,
                "currency": str(tx["currency"]),
                "amount_cents": int(tx["amount_cents"]),
                "status": status,
                "reason": reason,
            }
        )

    # Batch-level reconciliation for bank batch_total entries
    bank_batch = bank_any[bank_any["entry_type"] == "batch_total"].copy()
    bank_batch_in_month_count = int(
        (bank_batch["settled_month"] == month).sum() if not bank_batch.empty else 0
    )

    # Platform charges with no per-tx bank settlement are attributed to the batch pool.
    # This is only unambiguous when exactly ONE batch_total entry exists in the month.
    # When multiple exist we cannot tell which charges belong to which batch without
    # additional provenance — we emit a warning in the reason.
    missing_from_per_tx = plat_m[
        (plat_m["type"] == "charge") & (~plat_m["tx_id"].astype(str).isin(bank_tx_id_set))
    ].copy()
    batch_pool_amount = int(missing_from_per_tx["amount_cents"].astype(int).sum())
    multi_batch_warning = (
        f"; WARNING: {bank_batch_in_month_count} batch_total entries in month — "
        "per-batch attribution is ambiguous"
        if bank_batch_in_month_count > 1
        else ""
    )

    batch_deltas: list[dict[str, Any]] = []
    for _, b in bank_batch.iterrows():
        batch_id = str(b["batch_id"])
        settled_month_b = str(b["settled_month"])
        if settled_month_b != month:
            continue
        bank_amount = int(b["amount_cents"])

        expected_amount = batch_pool_amount
        delta = bank_amount - expected_amount

        batch_status: Status
        batch_reason: str
        if expected_amount == 0:
            batch_status = "unmatched_bank_entry"
            batch_reason = "batch_total entry but no inferred platform items" + multi_batch_warning
        elif abs(delta) <= rounding_threshold_cents and delta != 0:
            batch_status = "rounding_batch_diff"
            batch_reason = (
                f"batch_total differs from inferred platform sum by {delta} cents"
                + multi_batch_warning
            )
        elif delta == 0:
            batch_status = "matched"
            batch_reason = "batch_total matches inferred platform sum" + multi_batch_warning
        else:
            batch_status = "unmatched_bank_entry"
            batch_reason = f"batch_total delta too large: {delta} cents" + multi_batch_warning

        batch_deltas.append(
            {
                "batch_id": batch_id,
                "settled_at": str(b["settled_at"]),
                "currency": str(b["currency"]),
                "bank_amount_cents": bank_amount,
                "inferred_platform_amount_cents": expected_amount,
                "delta_cents": int(delta),
                "status": batch_status,
                "reason": batch_reason,
            }
        )
        report_rows.append(
            {
                "entity": "bank_batch",
                "month": month,
                "tx_id": "",
                "bank_id": str(b["bank_id"]),
                "batch_id": batch_id,
                "created_at": "",
                "settled_at": str(b["settled_at"]),
                "currency": str(b["currency"]),
                "amount_cents": bank_amount,
                "status": batch_status,
                "reason": batch_reason,
            }
        )

    # Bank-side unmatched per_tx entries settled in month with no platform match
    bank_settled_m = bank_per_tx[bank_per_tx["settled_month"] == month].copy()
    plat_tx_id_set = set(plat_m["tx_id"].astype(str).tolist())
    for _, b in bank_settled_m.iterrows():
        t = str(b["tx_id"])
        if t == "" or t in plat_tx_id_set:
            continue
        report_rows.append(
            {
                "entity": "bank_entry",
                "month": month,
                "tx_id": t,
                "bank_id": str(b["bank_id"]),
                "batch_id": str(b["batch_id"]),
                "created_at": "",
                "settled_at": str(b["settled_at"]),
                "currency": str(b["currency"]),
                "amount_cents": int(b["amount_cents"]),
                "status": "unmatched_bank_entry",
                "reason": "bank per_tx settled in month but no platform tx in month",
            }
        )

    report_df = pd.DataFrame(report_rows)

    # Summary aggregates
    counts = report_df.groupby(["entity", "status"]).size().reset_index(name="count")
    totals = (
        report_df.groupby(["entity", "status"])["amount_cents"]
        .sum()
        .reset_index(name="amount_cents_sum")
    )
    summary = {
        "month": month,
        "counts": counts.to_dict(orient="records"),
        "totals": totals.to_dict(orient="records"),
        "platform_duplicates": sorted(list(plat_dup_tx_ids)),
        "bank_duplicate_tx_ids": sorted(list(bank_dup_tx_ids)),
        "bank_duplicate_bank_ids": sorted(list(bank_dup_bank_ids)),
        "batch_deltas": batch_deltas,
    }

    return ReconResult(report_df=report_df, summary=summary)


def write_outputs(
    result: ReconResult,
    *,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = out_dir / "recon_report.csv"
    summary_json = out_dir / "recon_summary.json"
    report_md = out_dir / "recon_report.md"

    result.report_df.sort_values(["entity", "status", "tx_id", "bank_id"]).to_csv(
        report_csv, index=False
    )
    summary_json.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")

    # Simple markdown report
    counts_df = pd.DataFrame(result.summary["counts"])
    if not counts_df.empty:
        counts_md = counts_df.sort_values(["entity", "count"], ascending=[True, False]).to_markdown(
            index=False
        )
    else:
        counts_md = "_(no rows)_"

    top_gaps = result.report_df[
        result.report_df["status"].isin(
            [
                "missing_settlement",
                "late_settlement",
                "bank_duplicate",
                "platform_duplicate",
                "refund_without_original",
                "rounding_batch_diff",
                "unmatched_bank_entry",
            ]
        )
    ].copy()
    top_gaps = top_gaps.head(25)
    top_md = top_gaps.to_markdown(index=False) if not top_gaps.empty else "_(none)_"

    report_md.write_text(
        "\n".join(
            [
                f"# Reconciliation report — {result.summary['month']}",
                "",
                "## Counts by status",
                "",
                counts_md,
                "",
                "## Example gaps (first 25 rows)",
                "",
                top_md,
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "report_csv": str(report_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }

