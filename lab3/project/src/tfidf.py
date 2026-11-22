"""Wlasna implementacja reprezentacji TF-IDF z normalizacja L2 (stdlib Pythona)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence


SparseVector = Dict[int, float]


class TfidfVectorizer:
    """Reprezentacja TF-IDF pojedynczych tokenow z odcieciem min_df i normalizacja L2.

    Wzor IDF (gladki): idf(t) = log((1 + N) / (1 + df(t))) + 1.0
    Wektor kazdego dokumentu ma norme L2 rowna 1.0 (o ile zawiera znane tokeny).
    """

    def __init__(self, *, min_df: int = 2) -> None:
        if min_df < 1:
            raise ValueError("min_df musi byc >= 1")
        self.min_df = int(min_df)
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self.n_docs_fit: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "TfidfVectorizer":
        """Buduje slownik i wagi IDF wylacznie na podstawie przekazanych dokumentow."""
        n_docs = len(tokenized_docs)
        if n_docs == 0:
            raise ValueError("Zbior dokumentow do fit() nie moze byc pusty")

        df_counts: Counter[str] = Counter()
        for tokens in tokenized_docs:
            for tok in set(tokens):
                df_counts[tok] += 1

        kept_terms = sorted(
            term for term, df in df_counts.items() if df >= self.min_df
        )
        self.vocab = {term: idx for idx, term in enumerate(kept_terms)}
        self.idf = [
            math.log((1.0 + n_docs) / (1.0 + df_counts[term])) + 1.0
            for term in kept_terms
        ]
        self.n_docs_fit = n_docs
        return self

    def transform_one(self, tokens: Sequence[str]) -> SparseVector:
        """Zamienia liste tokenow na rzadki wektor TF-IDF (nieznane tokeny sa pomijane)."""
        if not self.vocab:
            return {}

        tf_counts: Counter[int] = Counter()
        vocab = self.vocab
        for tok in tokens:
            idx = vocab.get(tok)
            if idx is not None:
                tf_counts[idx] += 1

        if not tf_counts:
            return {}

        idf = self.idf
        raw: Dict[int, float] = {}
        sq_sum = 0.0
        for idx, count in tf_counts.items():
            val = float(count) * idf[idx]
            raw[idx] = val
            sq_sum += val * val

        norm = math.sqrt(sq_sum)
        if norm == 0.0:
            return {}
        return {idx: val / norm for idx, val in sorted(raw.items())}

    def transform(self, tokenized_docs: Sequence[Sequence[str]]) -> List[SparseVector]:
        """Zamienia kolekcje stokenizowanych dokumentow na liste wektorow TF-IDF."""
        return [self.transform_one(tokens) for tokens in tokenized_docs]

    def fit_transform(
        self, tokenized_docs: Sequence[Sequence[str]]
    ) -> List[SparseVector]:
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)

    def index_to_term(self) -> List[str]:
        terms = [""] * len(self.vocab)
        for term, idx in self.vocab.items():
            terms[idx] = term
        return terms

    def to_dict(self) -> Dict[str, object]:
        return {
            "min_df": self.min_df,
            "norm": "l2",
            "smooth_idf": True,
            "n_docs_fit": self.n_docs_fit,
            "terms": self.index_to_term(),
            "idf": list(self.idf),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TfidfVectorizer":
        vec = cls(min_df=int(data["min_df"]))
        terms = list(data["terms"])  # type: ignore[arg-type]
        idf = [float(v) for v in data["idf"]]  # type: ignore[arg-type]
        if len(terms) != len(idf):
            raise ValueError("Niezgodna dlugosc 'terms' i 'idf' w artefakcie TF-IDF")
        vec.vocab = {term: idx for idx, term in enumerate(terms)}
        vec.idf = idf
        vec.n_docs_fit = int(data.get("n_docs_fit", 0))
        return vec
