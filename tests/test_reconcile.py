from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.generate_data import generate_synthetic_data
from src.reconcile import load_bank_csv, load_platform_csv, reconcile_month, write_outputs


def _norm_via_csv(tmp_path: Path, platform_df: pd.DataFrame, bank_df: pd.DataFrame):
    plat_path = tmp_path / "platform_transactions.csv"
    bank_path = tmp_path / "bank_settlements.csv"
    platform_df.to_csv(plat_path, index=False)
    bank_df.to_csv(bank_path, index=False)
    return load_platform_csv(plat_path), load_bank_csv(bank_path)


def test_determinism_same_seed_same_outputs(tmp_path: Path):
    p1, b1, planted1 = generate_synthetic_data(month="2026-02", seed=7)
    p2, b2, planted2 = generate_synthetic_data(month="2026-02", seed=7)

    assert planted1 == planted2
    assert p1.equals(p2)
    assert b1.equals(b2)

    plat_norm, bank_norm = _norm_via_csv(tmp_path, p1, b1)
    r1 = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")
    r2 = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")
    assert r1.summary == r2.summary


def test_all_amounts_are_int_cents(tmp_path: Path):
    platform_df, bank_df, _ = generate_synthetic_data(month="2026-02", seed=7)
    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)

    assert pd.api.types.is_integer_dtype(plat_norm["amount_cents"])
    assert pd.api.types.is_integer_dtype(bank_norm["amount_cents"])


def test_late_settlement_detected(tmp_path: Path):
    platform_df, bank_df, planted = generate_synthetic_data(month="2026-02", seed=7)
    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    row = result.report_df[(result.report_df["entity"] == "platform_tx") & (result.report_df["tx_id"] == planted.late_settlement_tx_id)]
    assert len(row) >= 1
    assert row.iloc[0]["status"] == "late_settlement"


def test_duplicate_detected_in_bank(tmp_path: Path):
    platform_df, bank_df, planted = generate_synthetic_data(month="2026-02", seed=7, duplicate_in="bank")
    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    assert planted.duplicate_bank_tx_id in set(result.summary["bank_duplicate_tx_ids"])
    tx_rows = result.report_df[(result.report_df["entity"] == "platform_tx") & (result.report_df["tx_id"] == planted.duplicate_bank_tx_id)]
    assert len(tx_rows) >= 1
    assert tx_rows.iloc[0]["status"] == "bank_duplicate"


def test_orphan_refund_detected(tmp_path: Path):
    platform_df, bank_df, planted = generate_synthetic_data(month="2026-02", seed=7)
    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    row = result.report_df[(result.report_df["entity"] == "platform_tx") & (result.report_df["tx_id"] == planted.orphan_refund_tx_id)]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "refund_without_original"
    assert planted.orphan_refund_original_tx_id in row.iloc[0]["reason"]


def test_rounding_batch_diff_reported(tmp_path: Path):
    platform_df, bank_df, planted = generate_synthetic_data(month="2026-02", seed=7)
    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    batch_rows = result.report_df[(result.report_df["entity"] == "bank_batch") & (result.report_df["batch_id"] == planted.rounding_batch_id)]
    assert len(batch_rows) == 1
    assert batch_rows.iloc[0]["status"] in {"rounding_batch_diff", "matched"}

    # If delta is non-zero, it must be classified as rounding_batch_diff
    deltas = [d for d in result.summary["batch_deltas"] if d["batch_id"] == planted.rounding_batch_id]
    assert len(deltas) == 1
    if deltas[0]["delta_cents"] != 0:
        assert deltas[0]["status"] == "rounding_batch_diff"


def test_amount_mismatch_detected(tmp_path: Path):
    """Bank settles a different amount for a valid tx_id → amount_mismatch, not matched."""
    platform_df, bank_df, _ = generate_synthetic_data(month="2026-02", seed=7)
    # Tamper: change one per_tx bank amount to be $1.00 (100 cents) higher.
    per_tx_mask = (bank_df["entry_type"] == "per_tx") & (bank_df["tx_id"] != "")
    first_idx = bank_df[per_tx_mask].index[0]
    tampered_tx_id = str(bank_df.loc[first_idx, "tx_id"])
    bank_df.loc[first_idx, "amount_cents"] = int(bank_df.loc[first_idx, "amount_cents"]) + 100

    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    row = result.report_df[
        (result.report_df["entity"] == "platform_tx")
        & (result.report_df["tx_id"] == tampered_tx_id)
    ]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "amount_mismatch"
    assert "100" in row.iloc[0]["reason"]


def test_currency_mismatch_detected(tmp_path: Path):
    """Bank settles in a different currency for a valid tx_id → currency_mismatch."""
    platform_df, bank_df, _ = generate_synthetic_data(month="2026-02", seed=7)
    per_tx_mask = (bank_df["entry_type"] == "per_tx") & (bank_df["tx_id"] != "")
    first_idx = bank_df[per_tx_mask].index[0]
    tampered_tx_id = str(bank_df.loc[first_idx, "tx_id"])
    bank_df.loc[first_idx, "currency"] = "EUR"

    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    row = result.report_df[
        (result.report_df["entity"] == "platform_tx")
        & (result.report_df["tx_id"] == tampered_tx_id)
    ]
    assert len(row) == 1
    assert row.iloc[0]["status"] == "currency_mismatch"
    assert "EUR" in row.iloc[0]["reason"]


def test_bank_duplicate_not_overridden_by_refund(tmp_path: Path):
    """A bank duplicate of a refund tx should still preserve bank_duplicate in status."""
    platform_df, bank_df, planted = generate_synthetic_data(month="2026-02", seed=7, duplicate_in="bank")
    # Manually turn the duplicated tx into a refund scenario by injecting a second
    # bank per_tx for the orphan refund tx (which has no matching original).
    orphan_row = bank_df[bank_df["tx_id"] == planted.orphan_refund_tx_id].iloc[0].to_dict()
    orphan_row["bank_id"] = "bank_202602_EXTRA_DUP"
    bank_df = pd.concat([bank_df, pd.DataFrame([orphan_row])], ignore_index=True)

    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    row = result.report_df[
        (result.report_df["entity"] == "platform_tx")
        & (result.report_df["tx_id"] == planted.orphan_refund_tx_id)
    ]
    assert len(row) == 1
    # bank_duplicate is higher priority and must be preserved; refund info appended to reason.
    assert row.iloc[0]["status"] == "bank_duplicate"
    assert "refund original_tx_id not found" in row.iloc[0]["reason"]


def test_generator_rejects_too_few_charges():
    with pytest.raises(ValueError, match="n_charges must be at least 15"):
        generate_synthetic_data(month="2026-02", seed=7, n_charges=5)


def test_multi_batch_warning_in_reason(tmp_path: Path):
    """When two batch_total entries exist in the month, reason contains the ambiguity warning."""
    platform_df, bank_df, planted = generate_synthetic_data(month="2026-02", seed=7)
    # Add a second batch_total entry for the same month.
    extra_batch = {
        "bank_id": "bank_202602_EXTRA_BATCH",
        "settled_at": "2026-02-15T11:00:00Z",
        "currency": "USD",
        "amount_cents": 999,
        "tx_id": "",
        "batch_id": "batch_202602_2026-02-15",
        "entry_type": "batch_total",
    }
    bank_df = pd.concat([bank_df, pd.DataFrame([extra_batch])], ignore_index=True)

    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    batch_rows = result.report_df[result.report_df["entity"] == "bank_batch"]
    assert len(batch_rows) == 2
    for reason in batch_rows["reason"].tolist():
        assert "WARNING" in reason


def test_write_outputs_creates_files(tmp_path: Path):
    platform_df, bank_df, _ = generate_synthetic_data(month="2026-02", seed=7)
    plat_norm, bank_norm = _norm_via_csv(tmp_path, platform_df, bank_df)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month="2026-02")

    out = write_outputs(result, out_dir=tmp_path / "out")
    assert Path(out["report_csv"]).exists()
    assert Path(out["summary_json"]).exists()
    assert Path(out["report_md"]).exists()

    # sanity: summary_json parses
    parsed = json.loads(Path(out["summary_json"]).read_text(encoding="utf-8"))
    assert parsed["month"] == "2026-02"

