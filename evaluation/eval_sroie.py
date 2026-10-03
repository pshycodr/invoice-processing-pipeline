"""Score the bill pipeline (preprocess -> OCR -> text model -> rules) on SROIE.

Dataset: jsdnrs/ICDAR2019-SROIE. It has no line items or categories, so only company, date and total are scored.
Run from the project root with Ollama running:
    uv run --extra eval python -m evaluation.eval_sroie --split test --n 100 --ocr glm --model gemma3:1b
"""
import argparse
import calendar
import csv
import json
import re
import statistics
import tempfile
import time
from datetime import date as _date
from difflib import SequenceMatcher
from pathlib import Path

from app import categories as category_store
from app import config
from app.pipeline.extract import read_text_bill
from app.pipeline.ocr import get_engine
from app.pipeline.preprocess import preprocess
from app.pipeline.validation import check

SETTINGS = {"ocr": config.DEFAULT_OCR, "model": config.DEFAULT_TEXT_MODEL}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}


def _valid(y, m, d):
    try:
        return _date(y, m, d).isoformat()
    except ValueError:
        return None


def _year(y):
    return 2000 + y if y < 100 else y


def date_candidates(s) -> set:
    """All plausible ISO dates for a string. Day-first and month-first are both returned."""
    s = (s or "").strip().lower()
    out = set()
    if not s:
        return out
    if m := re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s):
        out.add(_valid(int(m[1]), int(m[2]), int(m[3])))
    if m := re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):
        out.add(_valid(int(m[1]), int(m[2]), int(m[3])))
    if m := re.search(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?!\d)", s):
        a, b, y = int(m[1]), int(m[2]), _year(int(m[3]))
        out.add(_valid(y, b, a))
        out.add(_valid(y, a, b))
    if (m := re.search(r"(\d{1,2})\s*([a-z]{3})[a-z]*\.?,?\s*(\d{2,4})", s)) and m[2] in MONTHS:
        out.add(_valid(_year(int(m[3])), MONTHS[m[2]], int(m[1])))
    if (m := re.search(r"([a-z]{3})[a-z]*\.?\s*(\d{1,2}),?\s*(\d{4})", s)) and m[1] in MONTHS:
        out.add(_valid(int(m[3]), MONTHS[m[1]], int(m[2])))
    out.discard(None)
    return out


def date_primary(s):
    """Best single reading of a ground-truth date: day-first, month-first only if day-first is impossible."""
    s = (s or "").strip().lower()
    if not s:
        return None
    if m := re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s):
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?!\d)", s):
        a, b, y = int(m[1]), int(m[2]), _year(int(m[3]))
        return _valid(y, b, a) or _valid(y, a, b)
    if (m := re.search(r"(\d{1,2})\s*([a-z]{3})[a-z]*\.?,?\s*(\d{2,4})", s)) and m[2] in MONTHS:
        return _valid(_year(int(m[3])), MONTHS[m[2]], int(m[1]))
    if (m := re.search(r"([a-z]{3})[a-z]*\.?\s*(\d{1,2}),?\s*(\d{4})", s)) and m[1] in MONTHS:
        return _valid(int(m[3]), MONTHS[m[1]], int(m[2]))
    return None


def parse_amount(s):
    if s is None:
        return None
    s = re.sub(r"(?i)rm|\$|,|\s", "", str(s))
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def norm_name(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())).strip()


def name_sim(a, b) -> float:
    a, b = norm_name(a), norm_name(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def ocr_has_company(gt, ocr_text, thr=0.8) -> bool:
    toks = norm_name(gt).split()
    hay = set(norm_name(ocr_text).split())
    return bool(toks) and sum(t in hay for t in toks) / len(toks) >= thr


def ocr_has_total(gt_total, ocr_text) -> bool:
    if gt_total is None:
        return False
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d{2}", ocr_text)]
    return any(abs(n - gt_total) < 0.005 for n in nums)


def run_one(row, workdir: Path) -> dict:
    gt = row["entities"]
    gt_total = parse_amount(gt.get("total"))
    rec = {
        "key": row["key"], "gt_company": gt.get("company", ""), "gt_date": gt.get("date", ""),
        "gt_total": gt_total, "status": "ok", "error": "", "engine_conf": None,
        "ocr_s": None, "llm_s": None, "pred_company": "", "pred_date": "", "pred_total": None,
        "n_items": None, "warnings": "", "ocr_text": "",
    }
    img_path = workdir / f"{row['key']}.jpg"
    row["image"].convert("RGB").save(img_path, quality=95)

    try:
        t0 = time.perf_counter()
        result = get_engine(SETTINGS["ocr"]).read(preprocess(img_path))
        text, conf = result.text, result.confidence
        rec["ocr_s"], rec["engine_conf"] = time.perf_counter() - t0, conf
        rec["ocr_text"] = text.replace("\n", " | ")
        if not text.strip():
            rec["status"], rec["error"] = "ocr_empty", "no text read"
            return rec
    except Exception as e:
        rec["status"], rec["error"] = "ocr_error", f"{type(e).__name__}: {e}"
        return rec

    rec["ocr_has_total"] = ocr_has_total(gt_total, text)
    rec["ocr_has_company"] = ocr_has_company(gt.get("company", ""), text)

    categories = category_store.load()
    try:
        t1 = time.perf_counter()
        bill, _ = read_text_bill(text, SETTINGS["model"], categories)
        rec["llm_s"] = time.perf_counter() - t1
    except Exception as e:
        rec["status"], rec["error"] = "llm_error", f"{type(e).__name__}: {str(e)[:200]}"
        return rec

    warnings = check(bill, categories)
    if conf < config.OCR_WARN_CONF:
        warnings.append("low OCR confidence")
    rec.update(pred_company=bill.store_name, pred_date=bill.date or "", pred_total=bill.total,
               n_items=len(bill.items), warnings="; ".join(warnings))
    return rec


def score(rec: dict) -> dict:
    """Add correctness flags. Failed samples count as wrong for every field."""
    ok = rec["status"] == "ok"
    rec["company_exact"] = ok and norm_name(rec["pred_company"]) == norm_name(rec["gt_company"])
    rec["company_ok"] = ok and name_sim(rec["pred_company"], rec["gt_company"]) >= 0.85
    gd = date_candidates(rec["gt_date"])
    rec["date_scored"] = bool(gd)
    rec["date_ok"] = ok and bool(gd & date_candidates(rec["pred_date"]))
    gp = date_primary(rec["gt_date"])
    rec["date_strict_ok"] = ok and gp is not None and rec["pred_date"] == gp
    rec["total_scored"] = rec["gt_total"] is not None
    rec["total_ok"] = (ok and rec["gt_total"] is not None and rec["pred_total"] is not None
                       and abs(rec["pred_total"] - rec["gt_total"]) < 0.005)
    rec["all_ok"] = rec["company_ok"] and rec["date_strict_ok"] and rec["total_ok"]
    rec["total_wrong_silent"] = ok and rec["total_scored"] and not rec["total_ok"] and not rec["warnings"]
    return rec


def pct(n, d):
    return f"{n}/{d} = {100 * n / d:.1f}%" if d else "n/a"


def summarize(recs: list) -> dict:
    N = len(recs)
    fails = [r for r in recs if r["status"] != "ok"]
    ok = [r for r in recs if r["status"] == "ok"]
    ts = [r for r in recs if r["total_scored"]]
    ds = [r for r in recs if r["date_scored"]]
    wrong_total = [r for r in ok if r["total_scored"] and not r["total_ok"]]

    def lat(key):
        v = sorted(r[key] for r in recs if r[key] is not None)
        if not v:
            return None
        return {"mean": round(statistics.mean(v), 2), "median": round(statistics.median(v), 2),
                "p90": round(v[int(0.9 * (len(v) - 1))], 2)}

    return {
        "samples": N,
        "failures": {"total": len(fails),
                     "ocr_empty": sum(r["status"] == "ocr_empty" for r in recs),
                     "ocr_error": sum(r["status"] == "ocr_error" for r in recs),
                     "llm_error": sum(r["status"] == "llm_error" for r in recs),
                     "failure_rate": round(len(fails) / N, 4) if N else None},
        "accuracy_all_samples(failures_count_wrong)": {
            "company_fuzzy>=0.85": pct(sum(r["company_ok"] for r in recs), N),
            "company_exact": pct(sum(r["company_exact"] for r in recs), N),
            "date_strict(day-first)": pct(sum(r["date_strict_ok"] for r in ds), len(ds)),
            "date_lenient(either d/m order)": pct(sum(r["date_ok"] for r in ds), len(ds)),
            "total": pct(sum(r["total_ok"] for r in ts), len(ts)),
            "all_three_fields": pct(sum(r["all_ok"] for r in recs), N)},
        "ocr_recall(upper_bound_for_llm)": {
            "gt_total_present_in_ocr": pct(sum(bool(r.get("ocr_has_total")) for r in ts), len(ts)),
            "gt_company_words_in_ocr": pct(sum(bool(r.get("ocr_has_company")) for r in recs), N)},
        "review_gate": {
            "wrong_totals": len(wrong_total),
            "wrong_totals_with_warning": sum(bool(r["warnings"]) for r in wrong_total),
            "silent_wrong_totals(no_warning)": sum(r["total_wrong_silent"] for r in wrong_total),
            "samples_with_any_warning": pct(sum(bool(r["warnings"]) for r in ok), len(ok))},
        "latency_s": {"ocr": lat("ocr_s"), "llm": lat("llm_s")},
    }


def load_rows(split, n, seed, stream):
    from datasets import load_dataset
    if stream:
        return load_dataset("jsdnrs/ICDAR2019-SROIE", split=split, streaming=True).take(n)
    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split=split)
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def evaluate(rows, out_dir: Path, verbose=True) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(rows, 1):
            rec = score(run_one(row, Path(tmp)))
            recs.append(rec)
            if verbose:
                print(f"[{i}] {rec['key']} {rec['status']:9s} company={'Y' if rec['company_ok'] else 'n'} "
                      f"date={'Y' if rec['date_ok'] else 'n'} total={'Y' if rec['total_ok'] else 'n'} "
                      f"gt_total={rec['gt_total']} pred_total={rec['pred_total']}", flush=True)
    with open(out_dir / "sroie_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in recs for k in r}))
        writer.writeheader()
        writer.writerows(recs)
    summary = summarize(recs)
    summary["config"] = {"ocr_engine": SETTINGS["ocr"], "text_model": SETTINGS["model"], "use_rules": config.USE_RULES,
                         "ocr_max_side": config.OCR_MAX_SIDE, "contrast": config.OCR_CONTRAST,
                         "sharpness": config.OCR_SHARPNESS}
    (out_dir / "sroie_summary.json").write_text(json.dumps(summary, indent=2))
    return recs, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", action="store_true", help="take the first N instead of downloading the whole split")
    ap.add_argument("--out", default="output/eval_out")
    ap.add_argument("--ocr", default=config.DEFAULT_OCR, choices=list(config.OCR_ENGINES))
    ap.add_argument("--model", default=config.DEFAULT_TEXT_MODEL, help="any Ollama model name")
    a = ap.parse_args()
    SETTINGS.update(ocr=a.ocr, model=a.model)
    print(f"[eval] ocr={a.ocr} model={a.model}")
    _, result = evaluate(load_rows(a.split, a.n, a.seed, a.stream), Path(a.out))
    print("\n" + json.dumps(result, indent=2))
    print(f"\nPer-receipt results: {a.out}/sroie_results.csv")
