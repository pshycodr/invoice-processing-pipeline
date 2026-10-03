"""Cross-check an eval run against SROIE ground truth that eval_sroie ignores: words, boxes and address.

Run from the project root after eval_sroie has written <out>/sroie_results.csv, with the same split, seed and --stream:
    uv run --extra eval python -m evaluation.verify --out output/eval_out --split test --stream
    uv run --extra eval python -m evaluation.verify --out output/eval_out --split test --stream --oracle

Checks:
1. Label sanity: are the GT total, company and date present in the GT words of the same receipt.
2. OCR quality: similarity to the GT words, plus recall of company and address tokens.
3. Rules on perfect text: the rules run on GT words rebuilt into lines (no OCR, no LLM).
4. Error attribution: wrong in the pipeline but right on GT words means OCR caused it.
   Wrong in both means the rules or the label caused it.
5. With --oracle: the full text-model extraction on GT text, which isolates the model from OCR.
"""
import argparse
import csv
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path

from app import categories as category_store
from app.pipeline.extract import read_text_bill
from app.pipeline.ocr import group_lines
from app.pipeline.rules import find_date, find_total, pick_store
from evaluation import eval_sroie as ev

FIELDS = ["company_exact", "company_ok", "date_strict_ok", "total_ok"]


def alnum(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def token_hit(ref, hyp, min_len=3):
    """Share of reference tokens found inside hyp with spaces removed, so glued OCR words still count."""
    toks = [t for t in re.findall(r"[A-Z0-9]+", (ref or "").upper()) if len(t) >= min_len]
    if not toks:
        return None
    h = alnum(hyp)
    return sum(t in h for t in toks) / len(toks)


def similarity(ref, hyp) -> float:
    a, b = alnum(ref), alnum(hyp)
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def gt_text(row) -> str:
    """Rebuild GT words into text lines with the same row grouping the OCR path uses."""
    res = [([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], w, 1.0)
           for w, (x1, y1, x2, y2) in zip(row["words"], row["bboxes"])]
    return "\n".join(group_lines(res))


def numbers(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d{2}", text)]


def mean(v):
    v = [x for x in v if x is not None]
    return round(statistics.mean(v), 3) if v else None


def tally(recs):
    ts = [r for r in recs if r["total_scored"]]
    ds = [r for r in recs if r["date_scored"]]
    n = len(recs)
    return {
        "company_fuzzy": ev.pct(sum(bool(r["company_ok"]) for r in recs), n),
        "company_exact": ev.pct(sum(bool(r["company_exact"]) for r in recs), n),
        "date_strict": ev.pct(sum(bool(r["date_strict_ok"]) for r in ds), len(ds)),
        "total": ev.pct(sum(bool(r["total_ok"]) for r in ts), len(ts)),
    }


def load_gt(split, n, seed, stream):
    from datasets import load_dataset
    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split=split, streaming=stream).remove_columns("image")
    if stream:
        return ds.take(n)
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder holding sroie_results.csv from eval_sroie")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="also run the text model on GT text (needs Ollama)")
    ap.add_argument("--model", default=ev.config.DEFAULT_TEXT_MODEL, help="Ollama model used by --oracle")
    a = ap.parse_args()
    out = Path(a.out)

    with open(out / "sroie_results.csv", newline="", encoding="utf-8") as f:
        run = list(csv.DictReader(f))
    gt = {r["key"]: r for r in load_gt(a.split, len(run), a.seed, a.stream)}
    categories = category_store.load()

    rows, pipe, rules_recs, oracle_recs, missing = [], [], [], [], []
    for r in run:
        g = gt.get(r["key"])
        if g is None:
            missing.append(r["key"])
            continue
        ent = g["entities"]
        gt_total = ev.parse_amount(ent.get("total"))
        text = gt_text(g)
        ocr = r["ocr_text"].replace(" | ", "\n")

        gd = ev.date_candidates(ent.get("date"))
        row = {
            "key": r["key"],
            "label_total_in_words": gt_total is not None and any(abs(x - gt_total) < 0.005 for x in numbers(text)),
            "label_company_in_words": ev.ocr_has_company(ent.get("company", ""), text),
            "label_date_in_words": bool(gd) and any(gd & ev.date_candidates(w) for w in g["words"]),
            "ocr_text_similarity": round(similarity(text, ocr), 3),
            "ocr_word_recall": token_hit(text, ocr),
            "ocr_company_recall": token_hit(ent.get("company", ""), ocr),
            "ocr_address_recall": token_hit(ent.get("address", ""), ocr),
            "n_items_pipeline": r["n_items"],
        }

        store, _ = pick_store(text)
        rec = {"status": "ok", "gt_company": ent.get("company", ""), "gt_date": ent.get("date", ""),
               "gt_total": gt_total, "pred_company": store or "", "pred_date": find_date(text) or "",
               "pred_total": find_total(text), "warnings": ""}
        ev.score(rec)
        rules_recs.append(rec)
        row.update({f"rules_gt_{k}": rec[k] for k in FIELDS})
        row.update(rules_gt_pred_company=rec["pred_company"], rules_gt_pred_date=rec["pred_date"],
                   rules_gt_pred_total=rec["pred_total"])

        p = {k: str(r[k]) == "True" for k in FIELDS + ["date_scored", "total_scored", "all_ok"]}
        pipe.append(p)
        for k in FIELDS:
            if not p[k]:
                row[f"blame_{k}"] = "ocr" if rec[k] else "rules_or_label"

        if a.oracle:
            orec = {"status": "ok", "gt_company": ent.get("company", ""), "gt_date": ent.get("date", ""),
                    "gt_total": gt_total, "pred_company": "", "pred_date": "", "pred_total": None, "warnings": ""}
            try:
                bill, _ = read_text_bill(text, a.model, categories)
                orec.update(pred_company=bill.store_name, pred_date=bill.date or "", pred_total=bill.total)
                row["oracle_n_items"] = len(bill.items)
            except Exception as e:
                orec["status"] = "llm_error"
                row["oracle_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            ev.score(orec)
            oracle_recs.append(orec)
            row.update({f"oracle_{k}": orec[k] for k in FIELDS})
        rows.append(row)

    n = len(rows)
    summary = {
        "matched_receipts": n,
        "missing_keys": missing,
        "label_sanity": {k: ev.pct(sum(bool(r[k]) for r in rows), n) for k in
                         ["label_total_in_words", "label_company_in_words", "label_date_in_words"]},
        "ocr_vs_gt_words": {k: mean([r[k] for r in rows]) for k in
                            ["ocr_text_similarity", "ocr_word_recall", "ocr_company_recall", "ocr_address_recall"]},
        "pipeline(from csv)": tally(pipe),
        "rules_on_gt_words": tally(rules_recs),
        "error_attribution": {k: {"ocr": [r["key"] for r in rows if r.get(f"blame_{k}") == "ocr"],
                                  "rules_or_label": [r["key"] for r in rows if r.get(f"blame_{k}") == "rules_or_label"]}
                              for k in FIELDS},
        "items": {"receipts_with_zero_items": ev.pct(sum(int(r["n_items_pipeline"] or 0) == 0 for r in rows), n)},
    }
    if a.oracle:
        summary["full_extract_on_gt_words"] = tally(oracle_recs)

    with open(out / "verify_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    (out / "verify_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nPer-receipt: {out}/verify_results.csv")


if __name__ == "__main__":
    main()
