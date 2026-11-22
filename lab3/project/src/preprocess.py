"""Przygotowanie tekstu: usuwanie artefaktow zrodla, tokeny specjalne i tokenizacja.

Wszystkie wzorce maskowania artefaktow zrodla zostaly ustalone wylacznie
na podstawie train.csv i plan-lab2.md (bez przegladania validation.csv).
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


# Staly prefiks tematu usuwany z poczatku pierwszej linii wiadomosci.
SUBJECT_PREFIX_PATTERN = r"^\s*Subject:\s*"

# Wzorce znacznikow list dyskusyjnych w linii tematu (ustalone wylacznie na train.csv).
SUBJECT_TAG_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        "subject_list_tags",
        r"\[(?:ILUG-Social|ILUG|Spambayes|zzzzteana|SAtalk|SAdev|s-t)\]",
    ),
)

# Wzorce stopek list dyskusyjnych i sladow anonimizacji korpusu (ustalone na train.csv).
BODY_MASKING_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # 1. Stopki ILUG / ILUG-Social (Mailman @ linux.ie), takze w cytowaniach '>'
    (
        "ilug_footer_header",
        r"(?:^|\n)(?:>\s*)*Irish Linux Users' Group(?:\s+Social Events)?:\s*(?:ilug|social)@linux\.ie[^\n]*",
    ),
    (
        "ilug_footer_listinfo",
        r"(?:^|\n)(?:>\s*)*https?://www\.linux\.ie/mailman/listinfo/(?:ilug|social)\s+for\s+\(un\)subscription(?:\s*\n(?:>\s*)*|\s+)information\.?",
    ),
    (
        "ilug_footer_maintainer",
        r"(?:^|\n)(?:>\s*)*List maintainer:\s*listmaster@linux\.ie[^\n]*",
    ),
    # 2. Stopki RPM-List (freshrpms.net)
    (
        "rpm_list_footer_name",
        r"(?:^|\n)(?:>\s*)*RPM-List mailing list[^\n]*",
    ),
    (
        "rpm_list_footer_url",
        r"(?:^|\n)(?:>\s*)*https?://lists\.freshrpms\.net/mailman/listinfo/rpm-list[^\n]*",
    ),
    # 3. Stopki Spamassassin-talk i blok sponsora OSDN (sourceforge.net)
    (
        "sf_osdn_sponsor_block",
        r"(?:^|\n)(?:>\s*)*-{10,}\s*\n(?:>\s*)*This sf\.net email is sponsored by:[\s\S]*?(?=\n(?:>\s*)*_{10,}|\n\s*\n|\Z)",
    ),
    (
        "satalk_footer_name",
        r"(?:^|\n)(?:>\s*)*Spamassassin-talk mailing list[^\n]*",
    ),
    (
        "satalk_footer_email",
        r"(?:^|\n)(?:>\s*)*Spamassassin-talk@lists\.sourceforge\.net[^\n]*",
    ),
    (
        "satalk_footer_url",
        r"(?:^|\n)(?:>\s*)*https?://lists\.sourceforge\.net/lists/listinfo/spamassassin-talk[^\n]*",
    ),
    # 4. Stopki Exmh-users / Exmh-workers (listman.redhat.com)
    (
        "exmh_footer_name",
        r"(?:^|\n)(?:>\s*)*Exmh-(?:users|workers) mailing list[^\n]*",
    ),
    (
        "exmh_footer_email",
        r"(?:^|\n)(?:>\s*)*Exmh-(?:users|workers)@redhat\.com[^\n]*",
    ),
    (
        "exmh_footer_url",
        r"(?:^|\n)(?:>\s*)*https?://listman\.redhat\.com/mailman/listinfo/exmh-(?:users|workers)[^\n]*",
    ),
    # 5. Baner i stopka Yahoo! Groups / egroups (lista zzzzteana / forteana)
    (
        "yahoo_groups_sponsor_box",
        r"-{5,}\s*Yahoo!\s*Groups\s*Sponsor\s*-+~-+>[\s\S]*?-{5,}~-+>",
    ),
    (
        "yahoo_groups_unsubscribe",
        r"(?:^|\n)(?:>\s*)*To unsubscribe from this group,\s*send an email to:\s*\n(?:>\s*)*forteana-unsubscribe@egroups\.com[^\n]*",
    ),
    (
        "yahoo_groups_terms",
        r"(?:^|\n)(?:>\s*)*Your use of Yahoo!\s*Groups is subject to\s+https?://docs\.yahoo\.com/info/terms/?[^\n]*",
    ),
    # 6. Ogolny blok stopki Mailman (separator '___' + '<Lista> mailing list' + listinfo)
    #    oraz jednoliniowa stopka listy FoRK (xent.com/mailman/listinfo/fork)
    (
        "generic_mailman_footer",
        r"(?:^|\n)(?:>\s*)*_{10,}\s*\n(?:>\s*)*[A-Za-z0-9._-]+\s+mailing list[^\n]*(?:\n(?:>\s*)*[^\n]*@[^\n]*)?\n(?:>\s*)*https?://[^\s]+/(?:mailman|lists)/listinfo/[^\s]+",
    ),
    (
        "fork_listinfo_footer",
        r"(?:^|\n)(?:>\s*)*https?://xent\.com/mailman/listinfo/fork[^\n]*",
    ),
    # 7. Metadane naglowkowe bramek RSS-to-email (URL: http://... oraz Date: Not supplied / RFC-2822 / ISO-8601)
    (
        "rss_url_header",
        r"(?:^|\n)(?:>\s*)*URL:\s*https?://[^\n]+",
    ),
    (
        "rss_date_header",
        r"(?:^|\n)(?:>\s*)*Date:\s*(?:Not supplied|(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+\w{3}\s+\d{4}[^\n]*|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\n]*)",
    ),
    # 7. Slady anonimizacji korpusu (zzzz, spamassassin.taint.org)
    (
        "anon_zzzz_taint_email",
        r"zzzz@spamassassin\.taint\.org",
    ),
    (
        "anon_zzzz_other_email",
        r"\bzzzz@[A-Za-z0-9._-]+\.[A-Za-z]{2,}",
    ),
    (
        "anon_spamassassin_taint_domain",
        r"(?:zzzz(?:_xy)?)?spamassassin\.taint\.org",
    ),
    (
        "anon_standalone_zzzz",
        r"\bzzzz\b",
    ),
)

_COMPILED_SUBJECT_PREFIX = re.compile(SUBJECT_PREFIX_PATTERN, flags=re.IGNORECASE)
_COMPILED_SUBJECT_TAGS = [
    (name, re.compile(pat, flags=re.IGNORECASE)) for name, pat in SUBJECT_TAG_PATTERNS
]
_COMPILED_BODY_MASK = [
    (name, re.compile(pat, flags=re.IGNORECASE)) for name, pat in BODY_MASKING_PATTERNS
]

# Wzorce tokenow zastepczych (krok 2.2 planu)
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>'\"()\[\]]+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE
)
# Kwoty pieniezne: prefiks ($14.95, US$ 100, £50, EUR 20) lub sufiks (100.000$, 50 dollars, 20 USD)
_MONEY_RE = re.compile(
    r"(?:(?:US\$|USD|EUR|GBP|[\$£€])\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:[\$£€]|USD|EUR|GBP|dollars?|cents?)\b)",
    flags=re.IGNORECASE,
)
_NUM_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_EXCLAM_RE = re.compile(r"!")
_TOKEN_RE = re.compile(r"<(?:caps|exclam|url|email|money|num)>|[a-z]{2,}")

_PLACEHOLDER_TO_TOKEN = {
    "__tok_url__": "<URL>",
    "__tok_email__": "<EMAIL>",
    "__tok_money__": "<MONEY>",
    "__tok_num__": "<NUM>",
    "__tok_caps__": "<CAPS>",
    "__tok_exclam__": "<EXCLAM>",
}


def get_preprocessing_spec(*, apply_masking: bool = True) -> Dict[str, object]:
    """Zwraca jawny, serializowalny opis regul przetwarzania w ustalonej kolejnosci."""
    return {
        "version": "1.0",
        "apply_masking": apply_masking,
        "steps": [
            "remove_subject_prefix",
            "remove_source_artifacts" if apply_masking else "skip_source_artifacts",
            "replace_url_email_money_num",
            "add_caps_and_replace_exclam",
            "lowercase_and_tokenize",
        ],
        "subject_prefix_pattern": SUBJECT_PREFIX_PATTERN,
        "subject_tag_patterns": [
            {"name": name, "pattern": pat} for name, pat in SUBJECT_TAG_PATTERNS
        ],
        "body_masking_patterns": [
            {"name": name, "pattern": pat} for name, pat in BODY_MASKING_PATTERNS
        ],
        "special_tokens": ["<CAPS>", "<EXCLAM>", "<URL>", "<EMAIL>", "<MONEY>", "<NUM>"],
        "min_word_len": 2,
    }


def mask_source_artifacts(text: str) -> str:
    """Usuwa prefiks Subject: oraz artefakty zrodla z kopii tekstu w pamieci."""
    cleaned = _COMPILED_SUBJECT_PREFIX.sub("", text, count=1)

    for _, rx in _COMPILED_SUBJECT_TAGS:
        cleaned = rx.sub(" ", cleaned)

    for _, rx in _COMPILED_BODY_MASK:
        cleaned = rx.sub(" ", cleaned)

    return cleaned


def tokenize_text(text: str, *, apply_masking: bool = True) -> List[str]:
    """Przetwarza surowy tekst wiadomosci na liste tokenow zgodnie z plan-lab2.md."""
    if apply_masking:
        work = mask_source_artifacts(text)
    else:
        work = _COMPILED_SUBJECT_PREFIX.sub("", text, count=1)

    # 1. Zastapienie adresow URL, e-mail, kwot i liczb znacznikami tymczasowymi
    #    (zeby adresy z wielkich liter lub liczby nie generowaly falszywych <CAPS>).
    work = _URL_RE.sub(" __tok_url__ ", work)
    work = _EMAIL_RE.sub(" __tok_email__ ", work)
    work = _MONEY_RE.sub(" __tok_money__ ", work)
    work = _NUM_RE.sub(" __tok_num__ ", work)

    # 2. Kazde slowo pisane w calosci wielkimi literami (>=3 litery) otrzymuje dodatkowy token <CAPS>,
    #    a kazdy wykrzyknik jest zamieniany na token <EXCLAM>.
    work = _CAPS_RE.sub(lambda m: f"{m.group(0)} __tok_caps__", work)
    work = _EXCLAM_RE.sub(" __tok_exclam__ ", work)

    # 3. Zamiana na male litery, przywrocenie kanonicznych tokenow specjalnych i tokenizacja.
    work = work.lower()
    for placeholder, canonical in _PLACEHOLDER_TO_TOKEN.items():
        work = work.replace(placeholder, canonical.lower())

    raw_tokens = _TOKEN_RE.findall(work)
    tokens: List[str] = []
    for tok in raw_tokens:
        if tok.startswith("<") and tok.endswith(">"):
            tokens.append(tok.upper())
        else:
            tokens.append(tok)
    return tokens


def tokenize_corpus(
    texts: Sequence[str], *, apply_masking: bool = True
) -> List[List[str]]:
    """Tokenizuje liste wiadomosci."""
    return [tokenize_text(t, apply_masking=apply_masking) for t in texts]
