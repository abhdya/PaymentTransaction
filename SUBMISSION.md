# Submission checklist (fill as you go)

## 1) Brainstorming thread (full LLM convo)
- Export the full chat log from Cursor/Claude Code and include it as-is.

## 2) Distilled prompt (single prompt)
Copy/paste the contents of `distilled_prompt.txt` (created in this repo) into your submission.

## 3) Execution thread (full run log, including wrong turns)
Capture terminal output of:
- installs
- running `python -m src.cli demo ...`
- running `pytest`

Tip (PowerShell):
```powershell
python -m src.cli demo --month 2026-02 --seed 7 *>&1 | Tee-Object -FilePath out\execution_log.txt
pytest -q *>&1 | Tee-Object -FilePath out\pytest_log.txt
```

## 4) Test cases
Run:
```powershell
pytest -q
```
The tests in `tests/` assert each planted gap type is detected and reported.

## 5) Working output
### Deployed link
- Deploy via Streamlit Community Cloud using `app/streamlit_app.py`.

### Code zip
- Zip the whole repo folder.

### Demo video
- Record: generate sample data → run reconciliation → show gap filters and a couple example rows.

### 3 sentences: what this would get wrong in production
1. If bank settlement references are missing or inconsistent, the heuristic matching may mis-associate transactions with the same amount within the same window.
2. True bank behavior includes fees, chargebacks, FX, and partial settlements; this demo’s simplified model can misclassify those as “unmatched” or “rounding” issues.
3. Duplicate detection here is rule-based; real systems need idempotency keys and stronger provenance to distinguish true duplicates from legitimate repeats (e.g., retries vs reversals).

