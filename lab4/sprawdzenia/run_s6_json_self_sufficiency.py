"""S6: Samowystarczalnosc plikow JSON (spam_classifier.json i alternative_model.json).

Porownuje sekcje 'preprocessing' w obu zapisanych plikach JSON z obecnymi stalymi
w lab3/project/src/preprocess.py, sprawdza brakujace wzorce tokenizacji, weryfikuje
zachowanie SpamClassifierArtifact.from_dict() (czy uzywa wzorcow z JSON) oraz zbiera
cytaty z README.md, summary.md i kodu dotyczace samowystarczalnosci JSON-a.
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s6_json_self_sufficiency.json"

sys.path.insert(0, str(LAB3_PROJECT))

import src.models as models_mod  # noqa: E402
import src.preprocess as prep_mod  # noqa: E402


def compare_model_preprocessing(model_path: Path) -> Dict[str, object]:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    json_prep = data["preprocessing"]
    code_spec = prep_mod.get_preprocessing_spec(
        apply_masking=bool(json_prep.get("apply_masking", True))
    )

    exact_spec_match = json_prep == code_spec

    # Szczegolowe porownanie sekcja po sekcji
    expected_subject_tags = [
        {"name": n, "pattern": p} for n, p in prep_mod.SUBJECT_TAG_PATTERNS
    ]
    expected_body_masks = [
        {"name": n, "pattern": p} for n, p in prep_mod.BODY_MASKING_PATTERNS
    ]

    field_checks = {
        "version_match": json_prep.get("version") == code_spec["version"],
        "apply_masking_match": json_prep.get("apply_masking") is True,
        "steps_order_match": json_prep.get("steps") == code_spec["steps"],
        "subject_prefix_match": (
            json_prep.get("subject_prefix_pattern") == prep_mod.SUBJECT_PREFIX_PATTERN
        ),
        "subject_tag_patterns_match": (
            json_prep.get("subject_tag_patterns") == expected_subject_tags
        ),
        "body_masking_patterns_match": (
            json_prep.get("body_masking_patterns") == expected_body_masks
        ),
        "special_tokens_match": json_prep.get("special_tokens") == code_spec["special_tokens"],
        "min_word_len_match": json_prep.get("min_word_len") == code_spec["min_word_len"],
        "n_subject_tag_patterns": len(json_prep.get("subject_tag_patterns", [])),
        "n_body_masking_patterns": len(json_prep.get("body_masking_patterns", [])),
    }

    # Test behawioralny w pamieci: czy zmiana wzorcow w JSON wplywa na from_dict() / predict_one()?
    mutated = copy.deepcopy(data)
    mutated["preprocessing"]["body_masking_patterns"] = []
    mutated["preprocessing"]["subject_tag_patterns"] = []
    mutated["preprocessing"]["subject_prefix_pattern"] = "NEVER_MATCH_XXX"
    mutated["preprocessing"]["min_word_len"] = 99

    art_orig = models_mod.SpamClassifierArtifact.from_dict(data)
    art_mut = models_mod.SpamClassifierArtifact.from_dict(mutated)

    probe_with_ilug = (
        "Subject: [ILUG] Test message\n\n"
        "Body text.\n\n"
        "Irish Linux Users' Group: ilug@linux.ie\n"
        "http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information.\n"
        "List maintainer: listmaster@linux.ie\n"
    )
    sc_orig, lbl_orig = art_orig.predict_one(probe_with_ilug)
    sc_mut, lbl_mut = art_mut.predict_one(probe_with_ilug)
    from_dict_ignores_json_patterns = (sc_orig == sc_mut) and (lbl_orig == lbl_mut)

    return {
        "model_file": model_path.relative_to(REPO_ROOT).as_posix(),
        "exact_match_with_get_preprocessing_spec": exact_spec_match,
        "field_checks": field_checks,
        "from_dict_ignores_mutated_json_patterns": from_dict_ignores_json_patterns,
        "mutation_probe_scores": {
            "original_json_score": sc_orig,
            "mutated_json_score": sc_mut,
            "original_label": lbl_orig,
            "mutated_label": lbl_mut,
        },
    }


def extract_doc_claims() -> List[Dict[str, object]]:
    """Zbiera konkretne linie z README.md, summary.md i src/ mowiace o JSON i regulach przetwarzania."""
    targets = [
        (LAB3_PROJECT / "README.md", [27, 28]),
        (LAB3_PROJECT / "results" / "summary.md", [5]),
        (LAB3_PROJECT / "src" / "models.py", [399, 400, 472, 473, 474, 475]),
        (LAB3_PROJECT / "src" / "predict.py", [1, 2, 3, 64]),
        (LAB3_PROJECT / "src" / "preprocess.py", [163, 164]),
    ]
    claims: List[Dict[str, object]] = []
    for path, lines_wanted in targets:
        file_lines = path.read_text(encoding="utf-8").splitlines()
        snippet = {
            ln: file_lines[ln - 1]
            for ln in lines_wanted
            if 1 <= ln <= len(file_lines)
        }
        claims.append(
            {
                "file": path.relative_to(REPO_ROOT).as_posix(),
                "lines": snippet,
            }
        )
    return claims


def main() -> None:
    main_res = compare_model_preprocessing(LAB3_PROJECT / "models" / "spam_classifier.json")
    alt_res = compare_model_preprocessing(LAB3_PROJECT / "models" / "alternative_model.json")

    # Regexy uzywane w tokenize_text(), ktorych NIE zapisano w sekcji 'preprocessing' JSON-a:
    tokenization_regexes_not_in_json = {
        "_URL_RE": prep_mod._URL_RE.pattern,
        "_EMAIL_RE": prep_mod._EMAIL_RE.pattern,
        "_MONEY_RE": prep_mod._MONEY_RE.pattern,
        "_NUM_RE": prep_mod._NUM_RE.pattern,
        "_CAPS_RE": prep_mod._CAPS_RE.pattern,
        "_EXCLAM_RE": prep_mod._EXCLAM_RE.pattern,
        "_TOKEN_RE": prep_mod._TOKEN_RE.pattern,
    }

    from_dict_source = inspect.getsource(models_mod.SpamClassifierArtifact.from_dict)

    both_identical_to_code = bool(
        main_res["exact_match_with_get_preprocessing_spec"]
        and alt_res["exact_match_with_get_preprocessing_spec"]
    )
    both_ignore_json_patterns = bool(
        main_res["from_dict_ignores_mutated_json_patterns"]
        and alt_res["from_dict_ignores_mutated_json_patterns"]
    )

    payload: Dict[str, object] = {
        "check": "S6",
        "description": "Porownanie sekcji preprocessing w obu plikach JSON z preprocess.py oraz sprawdzenie samowystarczalnosci from_dict()",
        "spam_classifier": main_res,
        "alternative_model": alt_res,
        "both_models_json_matches_preprocess_py": both_identical_to_code,
        "both_models_from_dict_ignores_json_patterns": both_ignore_json_patterns,
        "tokenization_regexes_missing_from_json": tokenization_regexes_not_in_json,
        "from_dict_source": from_dict_source,
        "documentation_and_code_claims": extract_doc_claims(),
        "classification_per_criteria": (
            "OGRANICZENIE (wplyw N) — zapisane pola JSON zgadzaja sie ze stalymi w preprocess.py, "
            "ale from_dict() ich nie wykonuje (czyta tylko apply_masking), a regexy tokenizacji nie sa zapisane w JSON."
            if both_identical_to_code
            else "POTWIERDZONY BLAD (wplyw W) — rozbieznosc miedzy JSON a preprocess.py."
        ),
        "passed": both_identical_to_code,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S6 zakonczone: json_matches_code={both_identical_to_code} | "
        f"from_dict_ignores_json_patterns={both_ignore_json_patterns} | "
        f"missing_tokenization_regexes={len(tokenization_regexes_not_in_json)}"
    )
    if not both_identical_to_code:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
