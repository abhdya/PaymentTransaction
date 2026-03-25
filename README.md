# OneLabAssessment — Month-end reconciliation

This repo generates synthetic payments data and reconciles **platform transactions** vs **bank settlements** at month end, highlighting where and why the books don’t balance.

## Assumptions
- Money is stored in **integer cents** (`amount_cents`) end-to-end (no floats).
- Platform records transactions instantly at `created_at`.
- Bank settles funds **1–2 days later** at `settled_at` (and may settle in the next month).
- Bank may produce **batched** settlement lines (no `tx_id`) that require **batch-level** reconciliation.
- Refunds reference an original transaction via `original_tx_id` (nullable). A refund with `original_tx_id` not found in platform data is a planted gap type.

## Data files
- `data/platform_transactions.csv`
  - `tx_id`, `created_at`, `type` (`charge`/`refund`), `currency`, `amount_cents`, `customer_id`, `original_tx_id`
- `data/bank_settlements.csv`
  - `bank_id`, `settled_at`, `currency`, `amount_cents`, `tx_id` (nullable), `batch_id`, `entry_type` (`per_tx`/`batch_total`)

Outputs go to `out/`:
- `out/recon_report.csv` (row-level classification)
- `out/recon_summary.json` (counts, totals, batch deltas)
- `out/recon_report.md` (human-readable summary)
