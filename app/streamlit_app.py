from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable regardless
# of which directory Streamlit is launched from.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

from src.generate_data import generate_synthetic_data
from src.reconcile import reconcile_month


st.set_page_config(page_title="Month-end Reconciliation Demo", layout="wide")

st.title("Month-end reconciliation (platform vs bank)")
st.write(
    "Generate synthetic data with planted gaps, reconcile, and inspect where/why the books don’t balance."
)

with st.sidebar:
    st.header("Inputs")
    month = st.text_input("Month (YYYY-MM)", value="2026-02")
    seed = st.number_input("Seed", min_value=0, value=7, step=1)
    n_customers = st.slider("Customers", min_value=5, max_value=200, value=50, step=5)
    n_charges = st.slider("Charges", min_value=50, max_value=2000, value=300, step=50)
    duplicate_in = st.selectbox("Duplicate planted in", options=["bank", "platform"], index=0)

    st.divider()
    st.subheader("Or upload CSVs")
    up_platform = st.file_uploader("platform_transactions.csv", type=["csv"])
    up_bank = st.file_uploader("bank_settlements.csv", type=["csv"])


def _read_uploaded_csv(upload) -> pd.DataFrame:
    data = upload.getvalue()
    return pd.read_csv(io.BytesIO(data))


if up_platform is not None and up_bank is not None:
    platform_df = _read_uploaded_csv(up_platform)
    bank_df = _read_uploaded_csv(up_bank)
    planted = None
    st.info("Using uploaded CSVs.")
else:
    platform_df, bank_df, planted = generate_synthetic_data(
        month=month,
        seed=int(seed),
        n_customers=int(n_customers),
        n_charges=int(n_charges),
        duplicate_in=str(duplicate_in),
    )
    st.caption("Using generated synthetic CSVs (not written to disk).")


col1, col2 = st.columns(2)
with col1:
    st.subheader("Platform transactions")
    st.dataframe(platform_df, use_container_width=True, height=300)
with col2:
    st.subheader("Bank settlements")
    st.dataframe(bank_df, use_container_width=True, height=300)

st.divider()

if st.button("Run reconciliation", type="primary"):
    # Normalize using the same logic as CSV loaders by writing to temp in-memory
    # (avoid file IO; rely on reconcile to compute month fields).
    # We mimic loader behavior for the dt/month columns.
    from src.reconcile import load_bank_csv, load_platform_csv

    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)
    plat_path = tmp_dir / "platform_transactions.csv"
    bank_path = tmp_dir / "bank_settlements.csv"
    platform_df.to_csv(plat_path, index=False)
    bank_df.to_csv(bank_path, index=False)

    plat_norm = load_platform_csv(plat_path)
    bank_norm = load_bank_csv(bank_path)
    result = reconcile_month(platform_df=plat_norm, bank_df=bank_norm, month=month)

    st.subheader("Summary (JSON)")
    st.code(json.dumps(result.summary, indent=2), language="json")

    st.subheader("Report (filterable)")
    report = result.report_df.copy()
    statuses = sorted(report["status"].dropna().unique().tolist())
    entities = sorted(report["entity"].dropna().unique().tolist())

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        status_sel = st.multiselect("Statuses", options=statuses, default=statuses)
    with fcol2:
        entity_sel = st.multiselect("Entities", options=entities, default=entities)

    filtered = report[report["status"].isin(status_sel) & report["entity"].isin(entity_sel)].copy()
    st.dataframe(filtered, use_container_width=True, height=450)

    st.subheader("Download outputs")
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered report CSV", data=csv_bytes, file_name="recon_report_filtered.csv")

    summary_bytes = json.dumps(result.summary, indent=2).encode("utf-8")
    st.download_button("Download summary JSON", data=summary_bytes, file_name="recon_summary.json")

    if planted is not None:
        st.subheader("Planted gap IDs (for demo)")
        st.json(planted.__dict__)

