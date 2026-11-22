"""S11: Analiza porownania 'z maskowaniem vs bez maskowania' i pozostalosci stopkowych w validation.

Realizuje:
1. Niezalezne zliczenie wiadomosci z >= 1 dopasowaniem wzorca maskowania (osobno ham i spam)
   w train.csv i validation.csv oraz rozbicie na kategorie sladow zrodla (tagi list, stopki list,
   naglowki RSS, slady anonimizacji).
2. Wariant bez maskowania (identyczny jak w diagnostics.py Labu 3): top 30 najsilniejszych wag
   spam i top 30 ham wraz z obiektywnym wyliczeniem dla kazdego tokenu, w ilu dokumentach train
   wystepowal przed maskowaniem, po maskowaniu oraz wylacznie w usuwanych sladach zrodla.
3. Dla dwoch pozostalosci stopkowych w ham validation (2d5d89669551e7d0 i 4848bd860e4e2f63):
   dokladny kontekst, tokeny fragmentu, ich obecnosc i wagi w spam_classifier.json oraz
   diagnostyczna zmiana oceny po usunieciu tej linii w pamieci.
4. Zapis wyniku w lab4/wyniki/s11_masking_claim.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
S2_DIR = REPO_ROOT / "lab4" / "proby" / "przebieg"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s11_masking_claim.json"

sys.path.insert(0, str(LAB3_PROJECT))

from src.cv_train import run_cv_and_train  # noqa: E402
from src.models import SpamClassifierArtifact  # noqa: E402
from src.preprocess import (  # noqa: E402
    BODY_MASKING_PATTERNS,
    SUBJECT_PREFIX_PATTERN,
    SUBJECT_TAG_PATTERNS,
    mask_source_artifacts,
    tokenize_corpus,
    tokenize_text,
)


def read_dataset(path: Path) -> Tuple[List[str], List[str], List[str]]:
    mids, texts, labels = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            mids.append(r["message_id"])
            texts.append(r["text"])
            labels.append(r["label"])
    return mids, texts, labels


def pattern_category(name: str) -> str:
    if name == "subject_list_tags":
        return "subject_list_tags"
    if name.startswith("rss_"):
        return "rss_headers"
    if name.startswith("anon_"):
        return "anonymization_traces"
    return "list_footers_and_banners"


def count_any_masking_hits(
    mids: Sequence[str], texts: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    all_patterns = list(SUBJECT_TAG_PATTERNS) + list(BODY_MASKING_PATTERNS)
    compiled = [(n, re.compile(p, flags=re.IGNORECASE)) for n, p in all_patterns]

    ham_total = sum(1 for l in labels if l == "ham")
    spam_total = sum(1 for l in labels if l == "spam")

    any_ham = 0
    any_spam = 0
    cat_hits: Dict[str, Dict[str, int]] = {
        "subject_list_tags": {"ham": 0, "spam": 0},
        "list_footers_and_banners": {"ham": 0, "spam": 0},
        "rss_headers": {"ham": 0, "spam": 0},
        "anonymization_traces": {"ham": 0, "spam": 0},
    }

    for mid, txt, lbl in zip(mids, texts, labels):
        work = re.sub(SUBJECT_PREFIX_PATTERN, "", txt, count=1, flags=re.IGNORECASE)
        matched_cats = set()
        for name, rx in compiled:
            if rx.search(work):
                matched_cats.add(pattern_category(name))
                work = rx.sub(" ", work)
        if matched_cats:
            if lbl == "ham":
                any_ham += 1
            else:
                any_spam += 1
            for c in matched_cats:
                cat_hits[c][lbl] += 1

    return {
        "n_total": len(labels),
        "n_ham": ham_total,
        "n_spam": spam_total,
        "any_pattern_ham": any_ham,
        "any_pattern_ham_pct": round(100.0 * any_ham / ham_total, 2),
        "any_pattern_spam": any_spam,
        "any_pattern_spam_pct": round(100.0 * any_spam / spam_total, 2),
        "any_pattern_total": any_ham + any_spam,
        "any_pattern_total_pct": round(100.0 * (any_ham + any_spam) / len(labels), 2),
        "by_category": {
            c: {
                "ham": v["ham"],
                "ham_pct": round(100.0 * v["ham"] / ham_total, 2),
                "spam": v["spam"],
                "spam_pct": round(100.0 * v["spam"] / spam_total, 2),
            }
            for c, v in cat_hits.items()
        },
    }


def analyze_unmasked_top_weights(
    tr_texts: Sequence[str], tr_labels: Sequence[str]
) -> Dict[str, object]:
    # Uruchomienie dokladnie tego samego wariantu bez maskowania co w diagnostics.py (w pamieci)
    unmasked_run = run_cv_and_train(S2_DIR, apply_masking=False, save_outputs=False)
    unmasked_art: SpamClassifierArtifact = unmasked_run["winner_artifact"]  # type: ignore[assignment]
    unmasked_cv = unmasked_run["cv_results"]
    dec = unmasked_cv["decision"]  # type: ignore[index]

    # Weryfikacja zgodnosci z diagnostics_section5.json z Labu 3
    diag_lab3 = json.loads(
        (LAB3_PROJECT / "results" / "diagnostics_section5.json").read_text(encoding="utf-8")
    )["b_masking_vs_no_masking"]["without_masking"]
    matches_lab3_diag = (
        dec["winner_model_type"] == diag_lab3["winner_model_type"]
        and dec["winner_param"] == diag_lab3["winner_param"]
        and unmasked_cv["train_summary"]["full_vocab_size"] == diag_lab3["vocab_size"]  # type: ignore[index]
    )

    # Dla kazdego dokumentu train policz wystapienia tokenow: bez maskowania vs z maskowaniem
    toks_unmasked = tokenize_corpus(tr_texts, apply_masking=False)
    toks_masked = tokenize_corpus(tr_texts, apply_masking=True)

    # Zliczenia per token: ile dokumentow (ham/spam) ma token przed maskowaniem, po maskowaniu,
    # oraz w ilu dokumentach liczba wystapien tokenu spadla po maskowaniu (czyli token siedzial w usuwanych sladach!)
    df_unm: Dict[str, Dict[str, int]] = {}
    df_msk: Dict[str, Dict[str, int]] = {}
    docs_reduced: Dict[str, Dict[str, int]] = {}
    docs_only_in_artifacts: Dict[str, Dict[str, int]] = {}

    for tu, tm, lbl in zip(toks_unmasked, toks_masked, tr_labels):
        cu = Counter(tu)
        cm = Counter(tm)
        for tok in cu:
            df_unm.setdefault(tok, {"ham": 0, "spam": 0})[lbl] += 1
            if cm[tok] < cu[tok]:
                docs_reduced.setdefault(tok, {"ham": 0, "spam": 0})[lbl] += 1
            if cm[tok] == 0:
                docs_only_in_artifacts.setdefault(tok, {"ham": 0, "spam": 0})[lbl] += 1
        for tok in cm:
            df_msk.setdefault(tok, {"ham": 0, "spam": 0})[lbl] += 1

    # Sprawdzenie takze w samych wycietych fragmentach (co dokladnie usuwaja 24 wzorce)
    all_patterns = list(SUBJECT_TAG_PATTERNS) + list(BODY_MASKING_PATTERNS)
    compiled = [(n, re.compile(p, flags=re.IGNORECASE)) for n, p in all_patterns]
    artifact_token_sources: Dict[str, Counter[str]] = {}
    for txt in tr_texts:
        work = re.sub(SUBJECT_PREFIX_PATTERN, "", txt, count=1, flags=re.IGNORECASE)
        for name, rx in compiled:
            for m in rx.finditer(work):
                frag_toks = set(tokenize_text("Subject: \n\n" + m.group(0), apply_masking=False))
                for t in frag_toks:
                    artifact_token_sources.setdefault(t, Counter())[name] += 1
            work = rx.sub(" ", work)

    top30 = unmasked_art.top_features(top_k=30)
    enriched: Dict[str, List[Dict[str, object]]] = {}
    artifact_driven_summary: Dict[str, List[Dict[str, object]]] = {}

    for side in ("ham", "spam"):
        side_list = []
        side_artifacts = []
        for rank, item in enumerate(top30[side], start=1):
            tok = str(item["token"])
            w = float(item["weight"])
            u_h = df_unm.get(tok, {}).get("ham", 0)
            u_s = df_unm.get(tok, {}).get("spam", 0)
            m_h = df_msk.get(tok, {}).get("ham", 0)
            m_s = df_msk.get(tok, {}).get("spam", 0)
            red_h = docs_reduced.get(tok, {}).get("ham", 0)
            red_s = docs_reduced.get(tok, {}).get("spam", 0)
            only_h = docs_only_in_artifacts.get(tok, {}).get("ham", 0)
            only_s = docs_only_in_artifacts.get(tok, {}).get("spam", 0)
            u_tot = u_h + u_s
            only_tot = only_h + only_s
            red_tot = red_h + red_s
            share_only_in_artifacts = (only_tot / u_tot) if u_tot > 0 else 0.0
            share_reduced = (red_tot / u_tot) if u_tot > 0 else 0.0
            pat_sources = dict(artifact_token_sources.get(tok, Counter()).most_common(5))

            # Token uznajemy za pochodzacy ze sladow zrodla (usuwanych przez 24 wzorce),
            # jesli >= 50% dokumentow z tym tokenem mialo go wylacznie w zamaskowanych sladach
            # (lub zniknal calkowicie / w przewazajacej czesci po maskowaniu).
            primarily_from_masked_artifacts = (only_tot >= 2) and (share_only_in_artifacts >= 0.5)

            entry = {
                "rank": rank,
                "token": tok,
                "weight": w,
                "df_unmasked": {"ham": u_h, "spam": u_s, "total": u_tot},
                "df_masked": {"ham": m_h, "spam": m_s, "total": m_h + m_s},
                "docs_only_in_masked_artifacts": {"ham": only_h, "spam": only_s, "total": only_tot},
                "docs_reduced_by_masking": {"ham": red_h, "spam": red_s, "total": red_tot},
                "share_only_in_artifacts": round(share_only_in_artifacts, 4),
                "share_reduced_by_masking": round(share_reduced, 4),
                "masking_pattern_sources": pat_sources,
                "primarily_from_masked_artifacts": primarily_from_masked_artifacts,
            }
            side_list.append(entry)
            if primarily_from_masked_artifacts or red_tot >= 5:
                side_artifacts.append(entry)
        enriched[side] = side_list
        artifact_driven_summary[side] = side_artifacts

    return {
        "matches_lab3_diagnostics_without_masking": matches_lab3_diag,
        "model_type": dec["winner_model_type"],
        "param": dec["winner_param"],
        "threshold": dec["winner_threshold"],
        "vocab_size": unmasked_cv["train_summary"]["full_vocab_size"],  # type: ignore[index]
        "top_30_enriched": enriched,
        "tokens_primarily_from_masked_artifacts": {
            "ham": [e for e in enriched["ham"] if e["primarily_from_masked_artifacts"]],
            "spam": [e for e in enriched["spam"] if e["primarily_from_masked_artifacts"]],
        },
        "tokens_with_any_major_artifact_overlap_ge_5_docs": artifact_driven_summary,
    }


def analyze_validation_remnants(
    va_mids: Sequence[str], va_texts: Sequence[str], va_labels: Sequence[str]
) -> List[Dict[str, object]]:
    art = SpamClassifierArtifact.load_json(LAB3_PROJECT / "models" / "spam_classifier.json")
    weights = art.model.feature_weights()
    vocab = art.vectorizer.vocab
    idf = art.vectorizer.idf

    targets = [
        ("2d5d89669551e7d0", re.compile(r"egroups", flags=re.IGNORECASE)),
        ("4848bd860e4e2f63", re.compile(r"(?im)^(?:>\s*)*[a-z0-9._-]+\s+mailing list\b")),
    ]

    results = []
    for mid, rx in targets:
        idx = va_mids.index(mid)
        raw_text = va_texts[idx]
        lbl = va_labels[idx]
        masked_text = mask_source_artifacts(raw_text)

        # Wyciagnij pasujace linie oraz krotki blok wokol trafienia
        lines = masked_text.splitlines()
        hit_line_indices = [i for i, ln in enumerate(lines) if rx.search(ln)]
        matched_lines = [lines[i] for i in hit_line_indices]

        # Zbuduj kontekst (od pierwszej pasujacej linii - 2 do konca wiadomosci, max 10 linii)
        if hit_line_indices:
            start = max(0, hit_line_indices[0] - 2)
            end = min(len(lines), hit_line_indices[-1] + 5)
            context_block = "\n".join(lines[start:end])
        else:
            context_block = ""

        # Tokeny z samych pasujacych linii (plus caly ewentualny blok stopki na koncu)
        matched_text_only = "\n".join(matched_lines)
        matched_tokens = sorted(set(tokenize_text("Subject: \n\n" + matched_text_only, apply_masking=False)))
        token_weights = []
        for tok in matched_tokens:
            if tok in vocab:
                j = vocab[tok]
                token_weights.append(
                    {"token": tok, "in_vocab": True, "weight": weights[j], "idf": idf[j]}
                )
            else:
                token_weights.append({"token": tok, "in_vocab": False, "weight": None, "idf": None})

        # Diagnostycznie: ocena oryginalna vs po usunieciu pasujacych linii z raw_text
        orig_score, orig_pred = art.predict_one(raw_text)
        raw_lines = raw_text.splitlines()
        cleaned_raw = "\n".join(ln for ln in raw_lines if not rx.search(ln))
        cleaned_score, cleaned_pred = art.predict_one(cleaned_raw)

        results.append(
            {
                "message_id": mid,
                "true_label": lbl,
                "matched_lines_in_masked_text": matched_lines,
                "context_snippet_in_masked_text": context_block,
                "matched_line_tokens_in_model_vocab": token_weights,
                "orig_score": orig_score,
                "orig_predicted_label": orig_pred,
                "score_after_removing_matched_lines": cleaned_score,
                "predicted_label_after_removing_matched_lines": cleaned_pred,
                "score_diff": cleaned_score - orig_score,
            }
        )
    return results


def main() -> None:
    tr_mids, tr_texts, tr_labels = read_dataset(LAB3_PROJECT / "data" / "train.csv")
    va_mids, va_texts, va_labels = read_dataset(LAB3_PROJECT / "data" / "validation.csv")

    train_hits = count_any_masking_hits(tr_mids, tr_texts, tr_labels)
    val_hits = count_any_masking_hits(va_mids, va_texts, va_labels)
    unmasked_top = analyze_unmasked_top_weights(tr_texts, tr_labels)
    remnants_detail = analyze_validation_remnants(va_mids, va_texts, va_labels)

    ham_art_tokens = [
        e["token"] for e in unmasked_top["tokens_primarily_from_masked_artifacts"]["ham"]  # type: ignore[index]
    ]
    spam_art_tokens = [
        e["token"] for e in unmasked_top["tokens_primarily_from_masked_artifacts"]["spam"]  # type: ignore[index]
    ]

    payload: Dict[str, object] = {
        "check": "S11",
        "description": "Analiza porownania z maskowaniem vs bez maskowania, proporcji sladow zrodla oraz 2 pozostalosci w validation",
        "masking_hit_rates": {
            "train": train_hits,
            "validation": val_hits,
        },
        "without_masking_top_30_analysis": unmasked_top,
        "validation_remnants_vocab_and_weights": remnants_detail,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S11 zakonczone | "
        f"train any_mask: ham={train_hits['any_pattern_ham']}/{train_hits['n_ham']} ({train_hits['any_pattern_ham_pct']}%), "
        f"spam={train_hits['any_pattern_spam']}/{train_hits['n_spam']} ({train_hits['any_pattern_spam_pct']}%) | "
        f"val any_mask: ham={val_hits['any_pattern_ham']}/{val_hits['n_ham']} ({val_hits['any_pattern_ham_pct']}%), "
        f"spam={val_hits['any_pattern_spam']}/{val_hits['n_spam']} ({val_hits['any_pattern_spam_pct']}%) | "
        f"unmasked top30 primarily_artifacts: ham={len(ham_art_tokens)} {ham_art_tokens}, spam={len(spam_art_tokens)} {spam_art_tokens}"
    )


if __name__ == "__main__":
    main()
