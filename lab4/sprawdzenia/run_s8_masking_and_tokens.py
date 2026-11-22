"""S8: Maskowanie i tokeny na danych (train.csv i validation.csv) + testy syntetyczne.

Realizuje:
  (i)   Liczbe dopasowan kazdego z 24 wzorcow maskowania w train i validation (spam/ham),
        sekwencyjnie (glowny) i osobno (standalone), plus diagnostyka pozostalosci stopek w validation.
  (ii)  W spamie validation (i dla porownania train): liczba wiadomosci z rdzeniem 'remov'
        i 'unsubscrib' przed i po mask_source_artifacts (niezaleznym regexem; oczekiwane: bez zmian).
  (iii) Testy syntetyczne i niezalezne zliczenie kwot z symbolem waluty PO liczbie
        (np. '100$', '50 €') w train i validation oraz ile z nich daje '<MONEY>' (osobno 'Euros').
  (iv)  Test syntetyczny i niezalezne zliczenie doslownych ciagow '<url>', '<money>', '<caps>',
        '<exclam>', '<email>', '<num>' w surowym tekscie train i validation.
  (dodatkowo diagnostycznie do tabeli d): URL zakonczony '!' oraz liczba tokenow w FN 017f0817d4c92951.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s8_masking_and_tokens.json"

sys.path.insert(0, str(LAB3_PROJECT))

from src.models import SpamClassifierArtifact  # noqa: E402
from src.preprocess import (  # noqa: E402
    BODY_MASKING_PATTERNS,
    SUBJECT_PREFIX_PATTERN,
    SUBJECT_TAG_PATTERNS,
    mask_source_artifacts,
    tokenize_text,
)


# Niezalezne regexy zdefiniowane w skrypcie sprawdzajacym (poza src/):
# (iii) Liczba z opcjonalnym separatorem tysiecy/dziesietnym i symbolem waluty PO liczbie,
#       ale bez poprzedzajacego symbolu waluty (zeby nie lapac np. "$100$").
IND_SUFFIX_CURRENCY_RE = re.compile(
    r"(?<![\$£€\d])\b\d[\d,]*(?:\.\d+)?\s*[\$£€]"
)
# Informacyjnie (osobno): slowna waluta 'euro'/'euros' lub 'pound'/'pounds' po liczbie
IND_WORD_EURO_POUND_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:euros?|pounds?)\b", flags=re.IGNORECASE
)
# (iv) Doslowne znaczniki przypominajace tokeny specjalne w surowym tekscie
IND_LITERAL_SPECIAL_TAG_RE = re.compile(
    r"<(?:caps|exclam|url|email|money|num)>", flags=re.IGNORECASE
)
# Bonus (tabela d): URL bezposrednio zakonczony wykrzyknikiem
IND_URL_ENDING_EXCLAM_RE = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>'\"()\[\]]*!", flags=re.IGNORECASE
)


def read_dataset(csv_path: Path) -> Tuple[List[str], List[str], List[str]]:
    mids, texts, labels = [], [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mids.append(row["message_id"])
            texts.append(row["text"])
            labels.append(row["label"])
    return mids, texts, labels


def pattern_stats(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> List[Dict[str, object]]:
    all_patterns = list(SUBJECT_TAG_PATTERNS) + list(BODY_MASKING_PATTERNS)
    work = [
        re.sub(SUBJECT_PREFIX_PATTERN, "", t, count=1, flags=re.IGNORECASE)
        for t in texts
    ]
    out: List[Dict[str, object]] = []
    for name, pat in all_patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        st_spam = [mid for mid, t, l in zip(mids, texts, labels) if l == "spam" and rx.search(t)]
        st_ham = [mid for mid, t, l in zip(mids, texts, labels) if l == "ham" and rx.search(t)]

        seq_spam: List[str] = []
        seq_ham: List[str] = []
        next_work: List[str] = []
        for mid, t, l in zip(mids, work, labels):
            if rx.search(t):
                if l == "spam":
                    seq_spam.append(mid)
                else:
                    seq_ham.append(mid)
                t = rx.sub(" ", t)
            next_work.append(t)
        work = next_work

        out.append(
            {
                "name": name,
                "sequential_spam": len(seq_spam),
                "sequential_ham": len(seq_ham),
                "sequential_total": len(seq_spam) + len(seq_ham),
                "sequential_spam_ids": seq_spam,
                "standalone_spam": len(st_spam),
                "standalone_ham": len(st_ham),
                "standalone_total": len(st_spam) + len(st_ham),
            }
        )
    return out


def check_validation_remnants(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    masked = [mask_source_artifacts(t) for t in texts]
    probes = [
        "groups sponsor",
        "yahoo! groups",
        "listinfo",
        "egroups",
        "sf.net email is sponsored",
        "list maintainer",
        "spamassassin.taint.org",
    ]
    remnants: Dict[str, List[Dict[str, str]]] = {}
    for p in probes:
        hits = [
            {"message_id": mid, "label": lbl}
            for mid, m, lbl in zip(mids, masked, labels)
            if p in m.lower()
        ]
        if hits:
            remnants[p] = hits
    # Sprawdz tez linie '<lista> mailing list' i naglowki RSS
    rx_probes = {
        "mailing_list_line": re.compile(r"(?im)^(?:>\s*)*[a-z0-9._-]+\s+mailing list\b"),
        "rss_url_line": re.compile(r"(?im)^(?:>\s*)*URL:\s*https?://"),
        "rss_date_not_supplied": re.compile(r"(?im)^(?:>\s*)*Date:\s*Not supplied\b"),
        "rss_date_iso8601": re.compile(r"(?im)^(?:>\s*)*Date:\s*\d{4}-\d{2}-\d{2}T"),
        "standalone_zzzz": re.compile(r"\bzzzz\b", flags=re.IGNORECASE),
    }
    for name, rx in rx_probes.items():
        hits = [
            {"message_id": mid, "label": lbl}
            for mid, m, lbl in zip(mids, masked, labels)
            if rx.search(m)
        ]
        if hits:
            remnants[name] = hits
    return {"n_probe_types_with_remnants": len(remnants), "remnants": remnants}


def check_stems(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    masked = [mask_source_artifacts(t) for t in texts]
    res: Dict[str, object] = {}
    for stem in ("remov", "unsubscrib"):
        rx = re.compile(stem, flags=re.IGNORECASE)
        before_ids = [mid for mid, t, l in zip(mids, texts, labels) if l == "spam" and rx.search(t)]
        after_ids = [mid for mid, m, l in zip(mids, masked, labels) if l == "spam" and rx.search(m)]
        res[stem] = {
            "spam_before_masking": len(before_ids),
            "spam_after_masking": len(after_ids),
            "unchanged": before_ids == after_ids,
            "lost_ids": sorted(set(before_ids) - set(after_ids)),
        }
    # Informacyjnie tez dokladne \bremove\b oraz token 'remove'
    rx_exact = re.compile(r"\bremove\b", flags=re.IGNORECASE)
    ex_before = sum(1 for t, l in zip(texts, labels) if l == "spam" and rx_exact.search(t))
    ex_after = sum(1 for m, l in zip(masked, labels) if l == "spam" and rx_exact.search(m))
    tok_masked = sum(
        1 for t, l in zip(texts, labels) if l == "spam" and "remove" in tokenize_text(t, apply_masking=True)
    )
    tok_unmasked = sum(
        1 for t, l in zip(texts, labels) if l == "spam" and "remove" in tokenize_text(t, apply_masking=False)
    )
    res["exact_remove_info"] = {
        "text_before": ex_before,
        "text_after": ex_after,
        "tokens_with_masking": tok_masked,
        "tokens_without_masking": tok_unmasked,
    }
    return res


def scan_suffix_currency(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    matches: List[Dict[str, object]] = []
    euro_pound_hits: List[Dict[str, object]] = []
    for mid, txt, lbl in zip(mids, texts, labels):
        found = IND_SUFFIX_CURRENCY_RE.findall(txt)
        if found:
            # Sprawdz dla kazdego znalezionego fragmentu osobno, czy tokenize_text daje <MONEY>
            per_fragment = [
                {
                    "fragment": frag,
                    "gives_money_token": "<MONEY>" in tokenize_text(f"Subject: x\n\n{frag}", apply_masking=True),
                    "tokens_for_fragment": tokenize_text(f"Subject: x\n\n{frag}", apply_masking=True),
                }
                for frag in found
            ]
            matches.append(
                {
                    "message_id": mid,
                    "label": lbl,
                    "n_occurrences": len(found),
                    "fragments": per_fragment,
                }
            )
        ep = IND_WORD_EURO_POUND_RE.findall(txt)
        if ep:
            euro_pound_hits.append({"message_id": mid, "label": lbl, "matches": ep})

    total_occ = sum(int(m["n_occurrences"]) for m in matches)
    occ_giving_money = sum(
        1
        for m in matches
        for frag in m["fragments"]  # type: ignore[union-attr]
        if frag["gives_money_token"]
    )
    return {
        "n_messages": len(matches),
        "n_spam_messages": sum(1 for m in matches if m["label"] == "spam"),
        "n_ham_messages": sum(1 for m in matches if m["label"] == "ham"),
        "n_occurrences": total_occ,
        "n_occurrences_giving_money_token": occ_giving_money,
        "n_occurrences_missed": total_occ - occ_giving_money,
        "messages": matches,
        "informational_word_euro_pound": {
            "n_messages": len(euro_pound_hits),
            "messages": euro_pound_hits,
        },
    }


def scan_literal_special_tags(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    hits: List[Dict[str, object]] = []
    for mid, txt, lbl in zip(mids, texts, labels):
        found = IND_LITERAL_SPECIAL_TAG_RE.findall(txt)
        if found:
            hits.append({"message_id": mid, "label": lbl, "matches": found})
    return {
        "n_messages": len(hits),
        "n_spam_messages": sum(1 for h in hits if h["label"] == "spam"),
        "n_ham_messages": sum(1 for h in hits if h["label"] == "ham"),
        "messages": hits,
    }


def scan_url_ending_exclam(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    hits: List[Dict[str, object]] = []
    for mid, txt, lbl in zip(mids, texts, labels):
        found = IND_URL_ENDING_EXCLAM_RE.findall(txt)
        if found:
            hits.append({"message_id": mid, "label": lbl, "n_matches": len(found), "examples": found[:3]})
    return {
        "n_messages": len(hits),
        "n_spam_messages": sum(1 for h in hits if h["label"] == "spam"),
        "n_ham_messages": sum(1 for h in hits if h["label"] == "ham"),
        "messages": hits,
    }


def main() -> None:
    tr_mids, tr_texts, tr_labels = read_dataset(LAB3_PROJECT / "data" / "train.csv")
    va_mids, va_texts, va_labels = read_dataset(LAB3_PROJECT / "data" / "validation.csv")

    # (i) Dopasowania 24 wzorcow na train i validation
    tr_pat_stats = pattern_stats(tr_mids, tr_texts, tr_labels)
    va_pat_stats = pattern_stats(va_mids, va_texts, va_labels)
    va_remnants = check_validation_remnants(va_mids, va_texts, va_labels)

    # (ii) Rdzenie remov / unsubscrib w spamie validation (i train)
    va_stems = check_stems(va_mids, va_texts, va_labels)
    tr_stems = check_stems(tr_mids, tr_texts, tr_labels)
    stems_validation_ok = bool(
        va_stems["remov"]["unchanged"] and va_stems["unsubscrib"]["unchanged"]  # type: ignore[index]
    )

    # (iii) Syntetyczne testy dla _MONEY_RE (symbol waluty PO liczbie) — oczekiwane przed uruchomieniem
    synthetic_money_cases = [
        {"input": "Subject: Offer 100$ now", "expected_contains": "<MONEY>"},
        {"input": "Subject: Win 100.000$ today", "expected_contains": "<MONEY>"},
        {"input": "Subject: Pay 50 € now", "expected_contains": "<MONEY>"},
        {"input": "Subject: Cost 25£ only", "expected_contains": "<MONEY>"},
    ]
    synthetic_money_results = []
    for case in synthetic_money_cases:
        toks = tokenize_text(case["input"], apply_masking=True)
        ok = case["expected_contains"] in toks
        synthetic_money_results.append(
            {
                "input": case["input"],
                "expected_contains": case["expected_contains"],
                "actual_tokens": toks,
                "matches_expected": ok,
            }
        )

    tr_suffix_currency = scan_suffix_currency(tr_mids, tr_texts, tr_labels)
    va_suffix_currency = scan_suffix_currency(va_mids, va_texts, va_labels)

    # (iv) Syntetyczny test dla doslownych ciagow '<url>', '<money>' — oczekiwane przed uruchomieniem
    synthetic_literal_tag_cases = [
        {
            "input": "Subject: See <url> and <money> in docs",
            "expected_not_contains": ["<URL>", "<MONEY>"],
        },
        {
            "input": "Subject: Token <CAPS> literal",
            "expected_not_contains": ["<CAPS>"],
        },
    ]
    synthetic_literal_tag_results = []
    for case in synthetic_literal_tag_cases:
        toks = tokenize_text(case["input"], apply_masking=True)
        leaked = [t for t in case["expected_not_contains"] if t in toks]
        synthetic_literal_tag_results.append(
            {
                "input": case["input"],
                "expected_not_contains": case["expected_not_contains"],
                "actual_tokens": toks,
                "spurious_special_tokens": leaked,
                "matches_expected": len(leaked) == 0,
            }
        )

    tr_literal_tags = scan_literal_special_tags(tr_mids, tr_texts, tr_labels)
    va_literal_tags = scan_literal_special_tags(va_mids, va_texts, va_labels)

    # Bonus diagnostyczny do tabeli (d):
    # 1. URL z wykrzyknikiem na koncu (syntetyczny + dane)
    url_exclam_syn_input = "Subject: Visit http://example.com! now"
    url_exclam_syn_tokens = tokenize_text(url_exclam_syn_input, apply_masking=True)
    tr_url_exclam = scan_url_ending_exclam(tr_mids, tr_texts, tr_labels)
    va_url_exclam = scan_url_ending_exclam(va_mids, va_texts, va_labels)

    # 2. Liczba znanych unikalnych tokenow w jedynym FN 017f0817d4c92951 (czy top_k=50 cokolwiek ucina)
    art = SpamClassifierArtifact.load_json(LAB3_PROJECT / "models" / "spam_classifier.json")
    fn_idx = va_mids.index("017f0817d4c92951")
    fn_text = va_texts[fn_idx]
    fn_all_contribs = art.explain_one(fn_text, top_k=10000)
    fn_neg_all = sorted(
        [c for c in fn_all_contribs if float(c["contribution"]) < 0.0],
        key=lambda c: float(c["contribution"]),
    )[:5]
    fn_top50 = art.explain_one(fn_text, top_k=50)
    fn_neg_from_50 = sorted(
        [c for c in fn_top50 if float(c["contribution"]) < 0.0],
        key=lambda c: float(c["contribution"]),
    )[:5]

    payload: Dict[str, object] = {
        "check": "S8",
        "description": "Maskowanie i tokeny na danych train.csv i validation.csv oraz testy syntetyczne",
        "part_i_masking_pattern_stats": {
            "train": tr_pat_stats,
            "validation": va_pat_stats,
            "validation_remnants_diagnostic": va_remnants,
        },
        "part_ii_stems_preservation": {
            "validation": va_stems,
            "train": tr_stems,
            "validation_stems_unchanged": stems_validation_ok,
        },
        "part_iii_suffix_currency_symbols": {
            "synthetic_tests": synthetic_money_results,
            "synthetic_bug_confirmed": any(
                not r["matches_expected"] for r in synthetic_money_results
            ),
            "train": tr_suffix_currency,
            "validation": va_suffix_currency,
        },
        "part_iv_literal_special_tags": {
            "synthetic_tests": synthetic_literal_tag_results,
            "synthetic_bug_confirmed": any(
                not r["matches_expected"] for r in synthetic_literal_tag_results
            ),
            "train": tr_literal_tags,
            "validation": va_literal_tags,
        },
        "extra_diagnostics_from_table_d": {
            "url_ending_with_exclam": {
                "synthetic_input": url_exclam_syn_input,
                "synthetic_expected_contains": ["<URL>", "<EXCLAM>"],
                "synthetic_actual_tokens": url_exclam_syn_tokens,
                "exclam_swallowed_in_synthetic": "<EXCLAM>" not in url_exclam_syn_tokens,
                "train": tr_url_exclam,
                "validation": va_url_exclam,
            },
            "fn_017f0817d4c92951_explain_top50_check": {
                "total_known_unique_tokens": len(fn_all_contribs),
                "exceeds_top_k_50": len(fn_all_contribs) > 50,
                "top5_negative_all_matches_top50": fn_neg_all == fn_neg_from_50,
                "top5_negative_all": fn_neg_all,
                "top5_negative_from_top50": fn_neg_from_50,
            },
        },
        "passed_stems_check": stems_validation_ok,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S8 zakonczone | stems_val_ok={stems_validation_ok} | "
        f"suffix_currency: syn_bug={payload['part_iii_suffix_currency_symbols']['synthetic_bug_confirmed']}, "
        f"train_msgs={tr_suffix_currency['n_messages']}, val_msgs={va_suffix_currency['n_messages']} | "
        f"literal_tags: syn_bug={payload['part_iv_literal_special_tags']['synthetic_bug_confirmed']}, "
        f"train_msgs={tr_literal_tags['n_messages']}, val_msgs={va_literal_tags['n_messages']} | "
        f"val_remnants={va_remnants['n_probe_types_with_remnants']} | "
        f"url_exclam: tr={tr_url_exclam['n_messages']}, va={va_url_exclam['n_messages']} | "
        f"fn_tokens={len(fn_all_contribs)} (top5_match={fn_neg_all == fn_neg_from_50})"
    )
    if not stems_validation_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
