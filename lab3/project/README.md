# Prototyp rozpoznawania spamu (Laboratorium 3)

Lokalny prototyp klasyfikacji wiadomości e-mail (`spam` / `ham`) na podstawie treści (tematu i tekstu wiadomości), realizujący plan z `../plan-lab2.md` na danych z korpusu Apache SpamAssassin ([opis danych](data/README.md)).

Prototyp działa wyłącznie lokalnie: nie łączy się ze skrzynką pocztową, nie używa sieci, nie otwiera linków z wiadomości i nie modyfikuje plików w `data/`.

## Wymagane zależności

- **Python 3.12** (w środowisku roboczym: `/Users/jasminetea/.pyenv/shims/python3`, przetestowano na `Python 3.12.1`).
- **Wyłącznie biblioteka standardowa Pythona** — brak zewnętrznych pakietów (`numpy`, `scipy`, `scikit-learn`, `joblib` nie są wymagane ani instalowane).

## Struktura projektu

- `data/` — niezmienione dane wejściowe z Labu 2 (`train.csv`, `validation.csv`, `manifest.json`, `README.md`).
- `src/` — kod prototypu:
  - `data_io.py` — odczyt CSV (w tym tryb bez czytania kolumny `label`) i weryfikacja `manifest.json`,
  - `preprocess.py` — maskowanie artefaktów źródła (24 wzorce wyprowadzone z `train.csv`), tokeny `<CAPS>`, `<EXCLAM>`, `<URL>`, `<EMAIL>`, `<MONEY>`, `<NUM>` i tokenizacja,
  - `tfidf.py` — własny `TfidfVectorizer` (`min_df=2`, gładzony IDF, normalizacja L2, serializacja JSON),
  - `models.py` — własna regresja logistyczna L2 (metoda Newtona z tożsamością Woodbury'ego i rozkładem Cholesky'ego), własny wielomianowy Naiwny Bayes oraz zapis/odczyt kompletnego artefaktu JSON,
  - `metrics.py` — macierz pomyłek, miary (FPR, czułość, precyzja, PR-AUC, dokładność) i reguła doboru progu,
  - `cv_train.py` — 5-krotna stratyfikowana walidacja krzyżowa na `train.csv` (ziarno `2026`), wybór modelu/ustawień/progu i uczenie końcowe,
  - `predict.py` — niezależne przewidywanie z zapisanego pliku JSON (bez czytania kolumny `label`, bez ponownego uczenia),
  - `evaluate.py` — jednorazowa ocena przewidywań względem `validation.csv` po `message_id` + punkty odniesienia + lista błędów FP/FN,
  - `diagnostics.py` — diagnostyka po ocenie (sekcja 5 planu: kalibracja CV, maskowanie vs brak maskowania, perturbacje na kopiach `validation`, wspólne wątki),
  - `run_all.py` — pełny przebieg od weryfikacji danych po wyniki i diagnostykę.
- `models/` — zapisane modele gotowe do ponownego użycia bez uczenia:
  - `models/spam_classifier.json` — wybrany model (`logistic_regression`, `C=10.0`, próg `0.328995`) wraz z regułami przetwarzania, słownikiem i wagami IDF,
  - `models/alternative_model.json` — model alternatywny (`multinomial_nb`, `alpha=0.1`, próg `0.580394`).
- `results/` — wyniki przebiegu i podsumowania:
  - `results/cv_results.json` — wyniki 5-krotnej CV dla wszystkich kandydatów, wyniki per fold, rozrzut progów i kalibracja OOF,
  - `results/train_top_weights.json` — 30 najsilniejszych wag `spam` i `ham` z `train`, statystyki dopasowań wzorców maskowania i tokeny jednoklasowe (`≥ 8` wiadomości),
  - `results/frozen_models_sha256.json` — sumy kontrolne SHA-256 modeli zapisane przed otwarciem `validation.csv`,
  - `results/validation_predictions.csv` — przewidywania wybranego modelu dla `validation.csv` (`message_id,score,predicted_label`),
  - `results/validation_predictions_alternative.csv` — przewidywania modelu alternatywnego (informacyjnie),
  - `results/validation_metrics.json` — macierz pomyłek, miary, punkty odniesienia i progi informacyjne (`0.25`, `0.40`, `0.50`),
  - `results/validation_errors.json` — lista wszystkich błędów FP i FN wraz z 5 słowami o największym wpływie,
  - `results/diagnostics_section5.json` — wyniki diagnostyki z sekcji 5 planu (a–d),
  - `results/summary.md` — zwięzłe podsumowanie wyników, odstępstw od planu, ograniczeń i niewykonanych sprawdzeń.
- `tests/` — testy jednostkowe, behawioralne i integracyjne (`unittest`).
- `proby/` — wyłącznie tymczasowe skrypty pomocnicze.

## Sposób uruchomienia

Wszystkie polecenia należy uruchamiać z katalogu `project/` przy użyciu Pythona 3.12 (np. `/Users/jasminetea/.pyenv/shims/python3`).

### 1. Pełny przebieg (dane → CV i uczenie → predykcja w osobnym procesie → ocena → diagnostyka)

```bash
/Users/jasminetea/.pyenv/shims/python3 src/run_all.py
```

Aby uruchomić predykcję, ocenę i diagnostykę z użyciem już zapisanych (zamrożonych) modeli w `models/` bez powtarzania uczenia:

```bash
/Users/jasminetea/.pyenv/shims/python3 src/run_all.py --skip-train
```

### 2. Przewidywanie dla pliku CSV bez ponownego uczenia (bez czytania kolumny `label`)

```bash
/Users/jasminetea/.pyenv/shims/python3 src/predict.py \
  --model models/spam_classifier.json \
  --input-csv data/validation.csv \
  --output-csv results/validation_predictions.csv
```

*(Plik wejściowy CSV wymaga wyłącznie kolumn `message_id,text`; jeśli zawiera kolumnę `label`, jest ona całkowicie ignorowana i nieodczytywana.)*

### 3. Przewidywanie dla pojedynczej wiadomości tekstowej (bez ponownego uczenia)

```bash
/Users/jasminetea/.pyenv/shims/python3 src/predict.py \
  --model models/spam_classifier.json \
  --text "Subject: FREE $500 CASH OFFER!! Click http://example.com to claim now!"
```

Można też przekazać treść przez standardowe wejście (`--stdin`):

```bash
printf "Subject: Kernel patch review\n\nPlease check the attached diff.\n" | \
  /Users/jasminetea/.pyenv/shims/python3 src/predict.py \
    --model models/spam_classifier.json \
    --stdin
```

### 4. Uruchomienie wszystkich testów (`unittest`)

```bash
/Users/jasminetea/.pyenv/shims/python3 -m unittest discover -s tests -t . -v
```
