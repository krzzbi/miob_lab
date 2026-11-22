"""Testy jednostkowe i behawioralne dla Czesci 1: wczytanie danych, maskowanie, TF-IDF."""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from src.data_io import compute_sha256, load_csv
from src.preprocess import (
    BODY_MASKING_PATTERNS,
    SUBJECT_TAG_PATTERNS,
    mask_source_artifacts,
    tokenize_corpus,
    tokenize_text,
)
from src.tfidf import TfidfVectorizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
MANIFEST_JSON = DATA_DIR / "manifest.json"


class TestPreprocessAndTfidf(unittest.TestCase):
    def test_hand_calculated_tokens_and_tfidf(self) -> None:
        """Sprawdza dokladne tokeny i recznie policzone wartosci TF-IDF na 3 dokumentach."""
        docs = [
            "Subject: FREE offer!! Visit https://example.com or a@b.com for $50 now 2026 OK",
            "Subject: free offer for 100 dollars!",
            "Subject: meeting notes 42",
        ]
        tokenized = tokenize_corpus(docs)

        # Dokument 0: 'Subject:' usuniete, FREE -> free + <CAPS>, '!!' -> 2x <EXCLAM>,
        # URL -> <URL>, email -> <EMAIL>, $50 -> <MONEY>, 2026 -> <NUM>, OK (2 litery) bez <CAPS>.
        self.assertEqual(
            tokenized[0],
            [
                "free",
                "<CAPS>",
                "offer",
                "<EXCLAM>",
                "<EXCLAM>",
                "visit",
                "<URL>",
                "or",
                "<EMAIL>",
                "for",
                "<MONEY>",
                "now",
                "<NUM>",
                "ok",
            ],
        )
        self.assertEqual(
            tokenized[1],
            ["free", "offer", "for", "<MONEY>", "<EXCLAM>"],
        )
        self.assertEqual(tokenized[2], ["meeting", "notes", "<NUM>"])

        # Reczne obliczenie TF-IDF dla min_df=2 (N = 3 dokumenty):
        # Slowa w >= 2 dokumentach:
        #   <EXCLAM>: df=2 (doc 0, 1)
        #   <MONEY>:  df=2 (doc 0, 1)
        #   <NUM>:    df=2 (doc 0, 2)
        #   for:      df=2 (doc 0, 1)
        #   free:     df=2 (doc 0, 1)
        #   offer:    df=2 (doc 0, 1)
        vec = TfidfVectorizer(min_df=2)
        matrix = vec.fit_transform(tokenized)

        expected_terms = ["<EXCLAM>", "<MONEY>", "<NUM>", "for", "free", "offer"]
        self.assertEqual(vec.index_to_term(), expected_terms)

        expected_idf = math.log((1.0 + 3.0) / (1.0 + 2.0)) + 1.0
        for idx in range(len(expected_terms)):
            self.assertAlmostEqual(vec.idf[idx], expected_idf, places=12)

        # W doc 0 wszystkie 6 slow maja identyczne IDF; tf = [2, 1, 1, 1, 1, 1].
        # Norma L2 = idf * sqrt(4 + 1 + 1 + 1 + 1 + 1) = idf * sqrt(9) = 3 * idf.
        # Zatem po normalizacji: <EXCLAM> = 2/3, pozostale 5 slow = 1/3.
        v0 = matrix[0]
        self.assertAlmostEqual(v0[vec.vocab["<EXCLAM>"]], 2.0 / 3.0, places=12)
        for term in ["<MONEY>", "<NUM>", "for", "free", "offer"]:
            self.assertAlmostEqual(v0[vec.vocab[term]], 1.0 / 3.0, places=12)
        l2_norm_0 = math.sqrt(sum(val * val for val in v0.values()))
        self.assertAlmostEqual(l2_norm_0, 1.0, places=12)

        # W doc 2 znane jest tylko <NUM>, wiec waga po normalizacji L2 wynosi 1.0.
        v2 = matrix[2]
        self.assertEqual(list(v2.keys()), [vec.vocab["<NUM>"]])
        self.assertAlmostEqual(v2[vec.vocab["<NUM>"]], 1.0, places=12)

    def test_tfidf_no_leakage_and_json_roundtrip(self) -> None:
        """Slownik i IDF pochodza tylko ze zbioru A; zapis/odczyt JSON zachowuje wektory."""
        train_docs = [["alpha", "beta"], ["alpha", "gamma"], ["beta", "gamma"]]
        new_docs = [["alpha", "unseenword", "unseenword", "unseenword"]]

        vec = TfidfVectorizer(min_df=2).fit(train_docs)
        self.assertNotIn("unseenword", vec.vocab)
        idf_before = list(vec.idf)
        transformed = vec.transform(new_docs)

        # Transformacja nie zmienia slownika ani IDF, a nieznane slowo jest ignorowane.
        self.assertNotIn("unseenword", vec.vocab)
        self.assertEqual(vec.idf, idf_before)
        self.assertEqual(list(transformed[0].keys()), [vec.vocab["alpha"]])
        self.assertAlmostEqual(transformed[0][vec.vocab["alpha"]], 1.0, places=12)

        # Round-trip JSON
        serialized = json.dumps(vec.to_dict(), sort_keys=True)
        restored = TfidfVectorizer.from_dict(json.loads(serialized))
        restored_vecs = restored.transform(new_docs)
        self.assertEqual(restored.vocab, vec.vocab)
        self.assertEqual(restored.idf, vec.idf)
        for idx, val in transformed[0].items():
            self.assertAlmostEqual(restored_vecs[0][idx], val, places=12)

    def test_load_csv_without_label_column(self) -> None:
        """load_csv(include_label=False) nie wymaga ani nie czyta kolumny label."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "unlabeled.csv"
            csv_path.write_text(
                'message_id,text\nm1,"Subject: Hello\n\nBody text"\n',
                encoding="utf-8",
            )
            mids, texts, labels = load_csv(csv_path, include_label=False)
            self.assertEqual(mids, ["m1"])
            self.assertEqual(len(texts), 1)
            self.assertIsNone(labels)

    def test_masking_preserves_spam_remove_and_removes_source_artifacts_on_train(
        self,
    ) -> None:
        """Sprawdzenie (c): maskowanie zachowuje 'remove' w spamie z train i usuwa artefakty."""
        mids, texts, labels = load_csv(TRAIN_CSV, include_label=True)
        assert labels is not None
        self.assertEqual(len(mids), 180)
        self.assertEqual(labels.count("ham"), 120)
        self.assertEqual(labels.count("spam"), 60)

        # Niezalezne zliczenie wiadomosci spam w train zawierajacych slowo 'remove' PRZED maskowaniem.
        spam_remove_before = sum(
            1
            for txt, lbl in zip(texts, labels)
            if lbl == "spam" and re.search(r"\bremove\b", txt, flags=re.IGNORECASE)
        )
        ham_unsub_before = sum(
            1
            for txt, lbl in zip(texts, labels)
            if lbl == "ham" and re.search(r"\bunsubscribe\b", txt, flags=re.IGNORECASE)
        )

        # Zliczenie PO maskowaniu (zarowno w zmaskowanym tekscie, jak i w tokenach
        # porownanych z tokenizacja bez maskowania, zeby oddzielic maskowanie od <URL>).
        masked_texts = [mask_source_artifacts(t) for t in texts]
        tokenized = tokenize_corpus(texts, apply_masking=True)
        tokenized_unmasked = tokenize_corpus(texts, apply_masking=False)

        spam_remove_after_text = sum(
            1
            for txt, lbl in zip(masked_texts, labels)
            if lbl == "spam" and re.search(r"\bremove\b", txt, flags=re.IGNORECASE)
        )
        spam_remove_after_tokens = sum(
            1
            for toks, lbl in zip(tokenized, labels)
            if lbl == "spam" and "remove" in toks
        )
        spam_remove_tokens_without_masking = sum(
            1
            for toks, lbl in zip(tokenized_unmasked, labels)
            if lbl == "spam" and "remove" in toks
        )
        ham_unsub_after_tokens = sum(
            1
            for toks, lbl in zip(tokenized, labels)
            if lbl == "ham" and "unsubscribe" in toks
        )

        # Wymaganie (c): maskowanie nie zmienia liczby wiadomosci spam z 'remove'
        # (24 -> 24 w tekscie; 17 -> 17 w tokenach po niezaleznej zamianie URL na <URL>).
        self.assertEqual(spam_remove_after_text, spam_remove_before)
        self.assertEqual(spam_remove_after_tokens, spam_remove_tokens_without_masking)
        # Wedlug plan-lab2.md po usunieciu stopek liczba 'unsubscribe' w ham wyraznie spada.
        self.assertLess(ham_unsub_after_tokens, ham_unsub_before)

        # Usuniecie wskazanych artefaktow zrodla w calym train.csv.
        forbidden_substrings = [
            "[spambayes]",
            "[satalk]",
            "[ilug-social]",
            "[ilug]",
            "[zzzzteana]",
            "spamassassin.taint.org",
            "listmaster@linux.ie",
            "forteana-unsubscribe@egroups.com",
            "lists.freshrpms.net/mailman/listinfo/rpm-list",
            "lists.sourceforge.net/lists/listinfo/spamassassin-talk",
            "listman.redhat.com/mailman/listinfo/exmh-",
        ]
        for mid, masked in zip(mids, masked_texts):
            low = masked.lower()
            for needle in forbidden_substrings:
                self.assertNotIn(
                    needle,
                    low,
                    msg=f"Artefakt {needle!r} pozostal w wiadomosci {mid}",
                )
            self.assertIsNone(
                re.search(r"\bzzzz\b", low),
                msg=f"Slad anonimizacji 'zzzz' pozostal w wiadomosci {mid}",
            )

        # Kontrola orientacyjna rozmiaru slownika na train (plan: ok. 2650-2900, tolerancja 2000-3500).
        vec = TfidfVectorizer(min_df=2).fit(tokenized)
        self.assertGreaterEqual(vec.vocab_size, 2000)
        self.assertLessEqual(vec.vocab_size, 3500)

        # Weryfikacja, ze wzorce maskowania sa jawnie zdefiniowane.
        self.assertGreater(len(SUBJECT_TAG_PATTERNS), 0)
        self.assertGreater(len(BODY_MASKING_PATTERNS), 0)

    def test_masking_pattern_coverage_remnants_and_stems_on_train(self) -> None:
        """Dodatkowe sprawdzenia maskowania (2a, 2b, 2c): pokrycie wzorcow, 0 pozostalosci, odmiany."""
        mids, texts, labels = load_csv(TRAIN_CSV, include_label=True)
        assert labels is not None

        # 2a: Kazdy wzorzec maskowania dopasowuje >= 1 wiadomosc w train (zarowno osobno, jak i sekwencyjnie).
        all_patterns = list(SUBJECT_TAG_PATTERNS) + list(BODY_MASKING_PATTERNS)
        work = [re.sub(r"^\s*Subject:\s*", "", t, count=1, flags=re.IGNORECASE) for t in texts]
        for name, pat in all_patterns:
            rx = re.compile(pat, flags=re.IGNORECASE)
            standalone_hits = sum(1 for t in texts if rx.search(t))
            seq_hits = 0
            next_work = []
            for t in work:
                if rx.search(t):
                    seq_hits += 1
                    t = rx.sub(" ", t)
                next_work.append(t)
            work = next_work
            self.assertGreater(
                standalone_hits,
                0,
                msg=f"Wzorzec {name!r} nie dopasowal zadnej wiadomosci w train (osobno)",
            )
            self.assertGreater(
                seq_hits,
                0,
                msg=f"Wzorzec {name!r} nie dopasowal zadnej wiadomosci w train (sekwencyjnie)",
            )

        # 2b: Po maskowaniu na train zostaje 0 pozostalosci stopek i banerow list.
        masked_texts = [mask_source_artifacts(t) for t in texts]
        footer_remnant_probes = [
            "groups sponsor",
            "yahoo! groups",
            "listinfo",
            "egroups",
            "sf.net email is sponsored",
            "list maintainer",
        ]
        for probe in footer_remnant_probes:
            remnants = [
                mid for mid, m in zip(mids, masked_texts) if probe in m.lower()
            ]
            self.assertEqual(
                len(remnants),
                0,
                msg=f"Pozostalosci {probe!r} po maskowaniu w wiadomosciach: {remnants}",
            )

        # Linie stopkowe '<NazwaListy> mailing list' powinny miec 0 pozostalosci
        # (z zachowaniem frazy 'mailing list' w zwyklej tresci spamu/dyskusji).
        footer_ml_rx = re.compile(
            r"(?im)^(?:>\s*)*[a-z0-9._-]+\s+mailing list\b"
        )
        footer_ml_remnants = [
            mid for mid, m in zip(mids, masked_texts) if footer_ml_rx.search(m)
        ]
        self.assertEqual(
            len(footer_ml_remnants),
            0,
            msg=f"Pozostalosci stopek '<Lista> mailing list': {footer_ml_remnants}",
        )

        # Pozostalosci naglowkow RSS (URL: i Date: Not supplied / ISO-8601) na poczatku linii = 0.
        rss_remnant_patterns = [
            ("URL:", re.compile(r"(?im)^(?:>\s*)*URL:\s*https?://")),
            ("Date: Not supplied", re.compile(r"(?im)^(?:>\s*)*Date:\s*Not supplied\b")),
            ("Date: ISO-8601", re.compile(r"(?im)^(?:>\s*)*Date:\s*\d{4}-\d{2}-\d{2}T")),
        ]
        for label_p, rx_p in rss_remnant_patterns:
            rss_hits = [mid for mid, m in zip(mids, masked_texts) if rx_p.search(m)]
            self.assertEqual(
                len(rss_hits),
                0,
                msg=f"Pozostalosci naglowka {label_p!r} po maskowaniu: {rss_hits}",
            )

        # 2c: Liczba wiadomosci spam w train pasujacych do 'remov' i 'unsubscrib'
        # (niezalezne zliczenie na surowym tekscie) nie zmienia sie po maskowaniu.
        for stem in ("remov", "unsubscrib"):
            rx = re.compile(stem, flags=re.IGNORECASE)
            before = sum(
                1 for t, lbl in zip(texts, labels) if lbl == "spam" and rx.search(t)
            )
            after = sum(
                1
                for m, lbl in zip(masked_texts, labels)
                if lbl == "spam" and rx.search(m)
            )
            self.assertEqual(
                after,
                before,
                msg=f"Maskowanie zmienilo liczbe spamow z {stem!r}: {before} -> {after}",
            )

    def test_train_csv_sha256_unchanged(self) -> None:
        """Sprawdza niezmiennosc train.csv wzgledem manifest.json (bez otwierania validation.csv)."""
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_sha = manifest["files"]["train.csv"]["sha256"]
        self.assertEqual(compute_sha256(TRAIN_CSV), expected_sha)


if __name__ == "__main__":
    unittest.main()
