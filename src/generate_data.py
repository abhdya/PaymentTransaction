from __future__ import annotations

import calendar
import dataclasses
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


UTC = timezone.utc


def _parse_month(month: str) -> tuple[int, int]:
    # month: "YYYY-MM"
    year_s, mon_s = month.split("-")
    year = int(year_s)
    mon = int(mon_s)
    if mon < 1 or mon > 12:
        raise ValueError("month must be YYYY-MM with 01..12")
    return year, mon


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    year, mon = _parse_month(month)
    _, last_day = calendar.monthrange(year, mon)
    start = datetime(year, mon, 1, 0, 0, 0, tzinfo=UTC)
    end_exclusive = datetime(year, mon, last_day, 23, 59, 59, tzinfo=UTC) + timedelta(
        seconds=1
    )
    return start, end_exclusive


def _rand_dt(rng: random.Random, start: datetime, end_exclusive: datetime) -> datetime:
    delta_s = int((end_exclusive - start).total_seconds())
    return start + timedelta(seconds=rng.randrange(delta_s))


def _next_month(month: str) -> str:
    year, mon = _parse_month(month)
    if mon == 12:
        return f"{year+1:04d}-01"
    return f"{year:04d}-{mon+1:02d}"


def _fmt_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclasses.dataclass(frozen=True)
class PlantedGaps:
    late_settlement_tx_id: str
    duplicate_bank_tx_id: str
    orphan_refund_tx_id: str
    orphan_refund_original_tx_id: str
    rounding_batch_id: str


def generate_synthetic_data(
    *,
    month: str,
    seed: int,
    n_customers: int = 50,
    n_charges: int = 300,
    duplicate_in: str = "bank",
    currency: str = "USD",
) -> tuple[pd.DataFrame, pd.DataFrame, PlantedGaps]:
    """
    Returns (platform_df, bank_df, planted_gaps).

    Guarantees the four required gap types exist:
    - transaction settling following month
    - rounding difference only at batch aggregate (bank has batch_total line off by a few cents)
    - duplicate entry (configurable; default in bank)
    - refund with no matching original transaction
    """
    if duplicate_in not in {"bank", "platform"}:
        raise ValueError("duplicate_in must be 'bank' or 'platform'")
    if n_charges < 15:
        raise ValueError("n_charges must be at least 15 (needed for rounding batch sample)")

    rng = random.Random(seed)
    start, end_exclusive = _month_bounds(month)
    next_month = _next_month(month)
    next_start, next_end_exclusive = _month_bounds(next_month)

    def tx_id(i: int) -> str:
        return f"tx_{month.replace('-', '')}_{i:06d}"

    customers = [f"cus_{i:04d}" for i in range(1, n_customers + 1)]

    platform_rows: list[dict] = []
    bank_rows: list[dict] = []

    # Create charges
    for i in range(1, n_charges + 1):
        created_at = _rand_dt(rng, start, end_exclusive)
        amount_cents = rng.randrange(150, 50_000)  # $1.50 .. $500
        customer_id = rng.choice(customers)
        platform_rows.append(
            {
                "tx_id": tx_id(i),
                "created_at": _fmt_iso(created_at),
                "type": "charge",
                "currency": currency,
                "amount_cents": int(amount_cents),
                "customer_id": customer_id,
                "original_tx_id": "",
            }
        )

    platform_df = pd.DataFrame(platform_rows)

    # Assign bank settlement dates 1-2 days later and batches by settled date.
    # We'll mostly create per-tx settlement lines (tx_id present).
    def settle_for(created_iso: str) -> datetime:
        created_dt = datetime.fromisoformat(created_iso.replace("Z", "+00:00"))
        lag_days = rng.choice([1, 2])
        # bank settles in daytime UTC
        settled = (created_dt + timedelta(days=lag_days)).replace(
            hour=rng.randrange(8, 18), minute=rng.randrange(0, 60), second=rng.randrange(0, 60)
        )
        return settled

    # Choose a tx to settle next month (late settlement gap)
    late_tx = platform_df.sort_values("created_at").iloc[-1]
    late_settlement_tx_id = str(late_tx["tx_id"])
    late_created_dt = datetime.fromisoformat(str(late_tx["created_at"]).replace("Z", "+00:00"))
    late_settled_dt = (late_created_dt + timedelta(days=2)).astimezone(UTC)
    if late_settled_dt < next_start:
        # force it into next month
        late_settled_dt = next_start + timedelta(hours=10)

    # Decide a batch that will be represented as a batch_total line with rounding delta.
    # We pick a settlement date within the month and group a subset of txs that settle that day.
    rounding_batch_date = _rand_dt(rng, start + timedelta(days=5), end_exclusive - timedelta(days=5)).date()
    rounding_batch_id = f"batch_{month.replace('-', '')}_{rounding_batch_date.isoformat()}"

    # Build bank per-tx settlements first (excluding those we will roll into the rounding batch_total)
    # We'll include a `batch_id` for every bank entry.
    rounding_tx_ids: set[str] = set()

    # Choose ~15 txs to be in the rounding batch (but keep them as platform charges)
    candidate_tx_ids = platform_df.sample(
        n=min(15, len(platform_df)), random_state=seed
    )["tx_id"].astype(str).tolist()
    rounding_tx_ids.update(candidate_tx_ids)

    # Ensure the late tx isn't in the rounding set to keep causes distinct.
    rounding_tx_ids.discard(late_settlement_tx_id)

    bank_id_counter = 1

    rounding_settled_dt = datetime(
        rounding_batch_date.year,
        rounding_batch_date.month,
        rounding_batch_date.day,
        11,
        0,
        0,
        tzinfo=UTC,
    )

    for _, row in platform_df.iterrows():
        this_tx_id = str(row["tx_id"])
        created_iso = str(row["created_at"])
        settled_dt = settle_for(created_iso)

        if this_tx_id == late_settlement_tx_id:
            settled_dt = late_settled_dt

        if this_tx_id in rounding_tx_ids:
            # These are rolled into a single batch_total entry. Force them to settle on the
            # batch date (synthetic gap: rounding only visible at batch aggregate).
            settled_dt = rounding_settled_dt
            continue

        batch_id = f"batch_{month.replace('-', '')}_{settled_dt.date().isoformat()}"
        bank_rows.append(
            {
                "bank_id": f"bank_{month.replace('-', '')}_{bank_id_counter:08d}",
                "settled_at": _fmt_iso(settled_dt),
                "currency": currency,
                "amount_cents": int(row["amount_cents"]),
                "tx_id": this_tx_id,
                "batch_id": batch_id,
                "entry_type": "per_tx",
            }
        )
        bank_id_counter += 1

    # Add a rounding batch_total entry: sum(platform txs in rounding batch) + small delta.
    # Delta is small and only visible at the aggregate level.
    rounding_txs_df = platform_df[platform_df["tx_id"].astype(str).isin(list(rounding_tx_ids))].copy()

    # The batch_total line is dated on rounding_settled_dt.
    rounding_sum = int(rounding_txs_df["amount_cents"].astype(int).sum())
    rounding_delta = rng.choice([-2, -1, 1, 2])

    bank_rows.append(
        {
            "bank_id": f"bank_{month.replace('-', '')}_{bank_id_counter:08d}",
            "settled_at": _fmt_iso(rounding_settled_dt),
            "currency": currency,
            "amount_cents": int(rounding_sum + rounding_delta),
            "tx_id": "",
            "batch_id": rounding_batch_id,
            "entry_type": "batch_total",
        }
    )
    bank_id_counter += 1

    bank_df = pd.DataFrame(bank_rows)

    # Plant: orphan refund (refund with no matching original)
    orphan_refund_original_tx_id = f"tx_{month.replace('-', '')}_ORPHAN_ORIG"
    orphan_refund_tx_id = f"rf_{month.replace('-', '')}_000001"
    refund_created_at = _rand_dt(rng, start + timedelta(days=2), end_exclusive - timedelta(days=2))
    refund_amount_cents = -int(rng.randrange(100, 10_000))

    platform_df = pd.concat(
        [
            platform_df,
            pd.DataFrame(
                [
                    {
                        "tx_id": orphan_refund_tx_id,
                        "created_at": _fmt_iso(refund_created_at),
                        "type": "refund",
                        "currency": currency,
                        "amount_cents": refund_amount_cents,
                        "customer_id": rng.choice(customers),
                        "original_tx_id": orphan_refund_original_tx_id,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # Bank also records the refund settlement (per-tx) to keep it reconcilable on bank side,
    # but the platform’s refund references a non-existent original.
    bank_df = pd.concat(
        [
            bank_df,
            pd.DataFrame(
                [
                    {
                        "bank_id": f"bank_{month.replace('-', '')}_{bank_id_counter:08d}",
                        "settled_at": _fmt_iso(refund_created_at + timedelta(days=1)),
                        "currency": currency,
                        "amount_cents": int(refund_amount_cents),
                        "tx_id": orphan_refund_tx_id,
                        "batch_id": f"batch_{month.replace('-', '')}_{(refund_created_at + timedelta(days=1)).date().isoformat()}",
                        "entry_type": "per_tx",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    bank_id_counter += 1

    # Plant: duplicate entry (in bank by default)
    duplicate_bank_tx_id = ""
    if duplicate_in == "bank":
        # pick a per_tx row with tx_id
        per_tx = bank_df[(bank_df["entry_type"] == "per_tx") & (bank_df["tx_id"].astype(str) != "")]
        dup_row = per_tx.sample(n=1, random_state=seed).iloc[0].to_dict()
        duplicate_bank_tx_id = str(dup_row["tx_id"])
        dup_row["bank_id"] = f"bank_{month.replace('-', '')}_{bank_id_counter:08d}"
        bank_id_counter += 1
        bank_df = pd.concat([bank_df, pd.DataFrame([dup_row])], ignore_index=True)
    else:
        dup_row = platform_df.sample(n=1, random_state=seed).iloc[0].to_dict()
        platform_df = pd.concat([platform_df, pd.DataFrame([dup_row])], ignore_index=True)

    # Ensure stable column order
    platform_df = platform_df[
        ["tx_id", "created_at", "type", "currency", "amount_cents", "customer_id", "original_tx_id"]
    ].copy()
    bank_df = bank_df[
        ["bank_id", "settled_at", "currency", "amount_cents", "tx_id", "batch_id", "entry_type"]
    ].copy()

    planted = PlantedGaps(
        late_settlement_tx_id=late_settlement_tx_id,
        duplicate_bank_tx_id=duplicate_bank_tx_id,
        orphan_refund_tx_id=orphan_refund_tx_id,
        orphan_refund_original_tx_id=orphan_refund_original_tx_id,
        rounding_batch_id=rounding_batch_id,
    )
    return platform_df, bank_df, planted


def write_csvs(
    platform_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    *,
    data_dir: Path,
) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    platform_path = data_dir / "platform_transactions.csv"
    bank_path = data_dir / "bank_settlements.csv"
    platform_df.to_csv(platform_path, index=False)
    bank_df.to_csv(bank_path, index=False)
    return platform_path, bank_path


def generate_and_write(
    *,
    month: str,
    seed: int,
    data_dir: Path,
    n_customers: int = 50,
    n_charges: int = 300,
    duplicate_in: str = "bank",
    currency: str = "USD",
) -> dict:
    platform_df, bank_df, planted = generate_synthetic_data(
        month=month,
        seed=seed,
        n_customers=n_customers,
        n_charges=n_charges,
        duplicate_in=duplicate_in,
        currency=currency,
    )
    platform_path, bank_path = write_csvs(platform_df, bank_df, data_dir=data_dir)
    return {
        "platform_path": str(platform_path),
        "bank_path": str(bank_path),
        "planted_gaps": dataclasses.asdict(planted),
        "counts": {
            "platform_rows": int(len(platform_df)),
            "bank_rows": int(len(bank_df)),
        },
    }

