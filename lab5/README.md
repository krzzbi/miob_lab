# Lab 5: Rozpoznawanie odręcznie zapisanych cyfr (8×8)

Lokalny prototyp klasyfikatora odręcznie zapisanych cyfr `0`–`9` dla pojedynczych obrazów PNG 8×8 px (8-bitowa skala szarości), zrealizowany w czystym Pythonie 3.12 (wyłącznie biblioteka standardowa).

Wybrany i zamrożony model końcowy: **k-NN (`k = 3`, metryka `L2`, głosy równe)**, nauczony wyłącznie na `data/train.csv` (600 obrazów) i zapisany w `model/model.json`.

---

## Środowisko

- **Python**: wymagany **Python 3.12** (sprawdzone na **Python 3.12.1**, macOS).
- **Zależności zewnętrzne**: brak — rozwiązanie korzysta wyłącznie z biblioteki standardowej Pythona (`zlib`, `struct`, `csv`, `json`, `hashlib`, `math`, `random`, `unittest` itd.); niczego nie trzeba instalować.
- W poleceniach poniżej użyto `python3` (upewnij się przez `python3 --version`, że wskazuje na Pythona 3.12). Zalecane jest uruchamianie z flagą `-B` oraz zmienną środowiskową `PYTHONDONTWRITEBYTECODE=1`, aby nie tworzyć katalogów `__pycache__`.

---

## Położenie i format danych

- Dane znajdują się w katalogu `lab5/data/` (szczegółowy opis pochodzenia, formatu i licencji: `data/README.md`, sumy kontrolne i pochodzenie rekordów: `data/manifest.json`).
- Pliki podziału: `data/train.csv` (600), `data/validation.csv` (200), `data/test.csv` (200). Kolumna `image_path` w plikach CSV zawiera ścieżki względne względem katalogu `data/` (np. `images/digit-17a28342e8c0.png`).
- Obrazy PNG znajdują się w `data/images/`.

---

## Zakres prototypu

Rozwiązanie przyjmuje wyłącznie **przygotowane, wycięte obrazy pojedynczych cyfr 8×8 px** w formacie zgodnym z danymi wejściowymi: PNG 8×8, 8-bitowa skala szarości (`color_type = 0`), bez przeplotu (`interlace = 0`), filtry wierszy `0`–`4`, poprawna suma CRC każdego chunku. Każdy inny format jest odrzucany z komunikatem błędu.

Prototyp nie obejmuje segmentacji całych formularzy, rozpoznawania zdjęć z telefonu ani skalowania dowolnych obrazów.

---

## Użycie (pojedynczy obraz PNG → cyfra)

Wszystkie polecenia uruchamiaj z katalogu `lab5/`. Przewidywanie działa z zapisanego pliku `model/model.json`, bez etykiety i bez ponownego uczenia:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B src/predict.py --model model/model.json --image data/images/digit-6380a7d3f106.png
```

Przykładowe wyjście (pojedyncza cyfra na stdout):

```text
8
```

---

## Weryfikacja oddanych wyników

Poniższe polecenia uruchamiaj bezpośrednio w oddanym katalogu `lab5/`. **Nie nadpisują one żadnego pliku w `wyniki/` ani w `model/`** — wszelkie pliki kontrolne zapisywane są wyłącznie do roboczego katalogu `proby/`.

- **Testy jednostkowe (w tym sprawdzenia dekodera S1 i S2)** — nie zapisuje żadnych plików w `wyniki/` ani `model/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

- **Porównanie SHA-256 danych (`data/`) z historycznym `wyniki/sha_data_przed.json`** — zapisuje wyłącznie do `proby/weryfikacja_sha_data.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_data_sha.py --data-dir data --output proby/weryfikacja_sha_data.json --compare wyniki/sha_data_przed.json
```

- **Sprawdzenie zamrożenia `model/model.json` i `src/` względem historycznego `wyniki/sha_przed_testem.json`** — zapisuje wyłącznie do `proby/weryfikacja_zamrozenia.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/verify_freeze.py --expected wyniki/sha_przed_testem.json --output proby/weryfikacja_zamrozenia.json
```

- **Niezależne macierze pomyłek S5 (bez importów z `src/`) dla `validation` i `test`** — zapisują wyłącznie do `proby/weryfikacja_s5_validation.json` i `proby/weryfikacja_s5_test.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_confusion_matrix.py --predictions wyniki/validation_predictions.csv --labels data/validation.csv --eval-json wyniki/validation_evaluation.json --output proby/weryfikacja_s5_validation.json
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_confusion_matrix.py --predictions wyniki/test_predictions.csv --labels data/test.csv --eval-json wyniki/test_evaluation.json --output proby/weryfikacja_s5_test.json
```

---

## Odtworzenie od zera (w czystej kopii)

> **Uwaga — nie uruchamiaj tej sekwencji w oddanym katalogu `lab5/`!**
> Poniższe polecenia nadpisują historyczne punkty odniesienia i model, w szczególności:
> - `wyniki/sha_data_przed.json` (przez `sprawdzenia/check_data_sha.py`),
> - `wyniki/sha_przed_testem.json` (przez `sprawdzenia/freeze_before_test.py`),
> - `model/model.json` (przez `src/train.py` oraz `sprawdzenia/check_s3_s4.py`),
> - pozostałe pliki raportów i przewidywań w `wyniki/`.
>
> Aby bezpiecznie odtworzyć całość od zera, skopiuj do osobnego pustego katalogu wyłącznie: `src/`, `sprawdzenia/`, `tests/`, `data/` i `README.md` (bez `model/`, `wyniki/` i `proby/`), wejdź do tej kopii i wykonaj polecenia w poniższej kolejności.

### 1. Testy jednostkowe (w tym sprawdzenia dekodera S1 i S2)
Nie zapisuje trwałych plików w kopii roboczej (używa katalogów tymczasowych).

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

### 2. Eksperyment wyboru modelu (`train.csv` 5-fold CV i `validation.csv`)
Orientacyjny czas wykonania: **ok. 3–4 minuty** na CPU (dominuje 5 konfiguracji softmax + 5-krotna CV dla najlepszego softmax).
Zapisuje/nadpisuje: `wyniki/eksperymenty_validation.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B src/experiment.py --learning-rate 1.0 --max-epochs 1500 --rel-tol 1e-5 --output wyniki/eksperymenty_validation.json
```

### 3. Uczenie i zapis modelu końcowego (wyłącznie na `train.csv`)
Zapisuje/nadpisuje: `model/model.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B src/train.py --model-type knn --k 3 --metric l2 --train-csv data/train.csv --images-dir data --output model/model.json
```

### 4. Przygotowanie plików CSV bez kolumny `label`
Skrypt `src/predict.py` celowo odrzuca pliki CSV zawierające kolumnę `label`.
Zapisuje/nadpisuje: `proby/validation_no_label.csv` oraz `proby/test_no_label.csv`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/make_no_label_csv.py --input data/validation.csv --output proby/validation_no_label.csv
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/make_no_label_csv.py --input data/test.csv --output proby/test_no_label.csv
```

### 5. Przewidywanie wsadowe dla `validation` i `test` (z zapisanego modelu)
Zapisuje/nadpisuje: `wyniki/validation_predictions.csv` oraz `wyniki/test_predictions.csv`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B src/predict.py --model model/model.json --input-csv proby/validation_no_label.csv --images-dir data --output-csv wyniki/validation_predictions.csv
PYTHONDONTWRITEBYTECODE=1 python3 -B src/predict.py --model model/model.json --input-csv proby/test_no_label.csv --images-dir data --output-csv wyniki/test_predictions.csv
```

### 6. Ocena przewidywań dla `validation` i `test`
Zapisuje/nadpisuje: `wyniki/validation_evaluation.json` oraz `wyniki/test_evaluation.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B src/evaluate.py --predictions wyniki/validation_predictions.csv --labels data/validation.csv --output wyniki/validation_evaluation.json
PYTHONDONTWRITEBYTECODE=1 python3 -B src/evaluate.py --predictions wyniki/test_predictions.csv --labels data/test.csv --output wyniki/test_evaluation.json
```

### 7. Sprawdzenia integralności, niezależne weryfikacje i analizy testowe

- **S3 (odtwarzalność bajtowa modelu) i S4 na `validation` (inferencja w osobnym procesie + odrzucenie kolumny `label`)**
  Zapisuje/nadpisuje: `model/model.json`, `proby/model_repro_check.json`, `proby/validation_no_label.csv`, `wyniki/validation_predictions.csv`, `wyniki/s3_s4_sprawdzenia.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_s3_s4.py
```

- **S4 na `test` (zgodność `wyniki/test_predictions.csv` z modelem w pamięci + 3 pojedyncze obrazy testowe)**
  Zapisuje/nadpisuje: `wyniki/s4_test.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_s4_test.py --output wyniki/s4_test.json
```

- **S5 (niezależna macierz pomyłek bez importów z `src/`, dla `validation` i `test`)**
  Zapisuje/nadpisuje: `wyniki/s5_niezalezna_macierz.json` oraz `wyniki/s5_test_macierz.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_confusion_matrix.py --predictions wyniki/validation_predictions.csv --labels data/validation.csv --eval-json wyniki/validation_evaluation.json --output wyniki/s5_niezalezna_macierz.json
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_confusion_matrix.py --predictions wyniki/test_predictions.csv --labels data/test.csv --eval-json wyniki/test_evaluation.json --output wyniki/s5_test_macierz.json
```

- **S7 (SHA-256 wszystkich plików w `data/` — w czystej kopii zapisuje punkt odniesienia i porównuje)**
  Zapisuje/nadpisuje: `wyniki/sha_data_przed.json` oraz `wyniki/sha_data_po_tescie.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_data_sha.py --data-dir data --output wyniki/sha_data_przed.json
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/check_data_sha.py --data-dir data --output wyniki/sha_data_po_tescie.json --compare wyniki/sha_data_przed.json
```

- **S8 i bramka zamrożenia (SHA-256 modelu oraz plików `src/` — w czystej kopii zapisuje nowy punkt odniesienia i weryfikuje)**
  Zapisuje/nadpisuje: `wyniki/sha_przed_testem.json` oraz `wyniki/sha_po_tescie_weryfikacja.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/freeze_before_test.py --model model/model.json --src-dir src --output wyniki/sha_przed_testem.json
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/verify_freeze.py --expected wyniki/sha_przed_testem.json --output wyniki/sha_po_tescie_weryfikacja.json
```

- **Punkt odniesienia (centroid na teście), dokładny test McNemara oraz analiza błędów testowych i wizualizacja PNG**
  Zapisuje/nadpisuje: `wyniki/test_odniesienie_centroid.json`, `wyniki/mcnemar_test.json`, `wyniki/bledy_test.json`, `wyniki/bledy_test.png`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/eval_centroid_test.py --train-csv data/train.csv --test-csv data/test.csv --images-dir data --output wyniki/test_odniesienie_centroid.json
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/mcnemar_test.py --output wyniki/mcnemar_test.json
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/analyze_test_errors.py --output-json wyniki/bledy_test.json --output-png wyniki/bledy_test.png
```

### 8. Opcjonalnie: odtworzenie badania zbieżności softmax (wyłącznie na `train.csv`)
Orientacyjny czas wykonania: **ok. 6–7 minut** na CPU (20 przebiegów: 4 kroki uczenia × 5 wartości `λ`).
Zapisuje/nadpisuje: `wyniki/softmax_zbieznosc_train.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B sprawdzenia/softmax_zbieznosc_train.py --output wyniki/softmax_zbieznosc_train.json
```

---

## Mapa plików

Katalog `proby/` służy wyłącznie jako roboczy brudnopis poza repozytorium (nie jest częścią oddawanych wyników końcowych).

| Plik wyjściowy | Polecenie / skrypt tworzący | Opis |
|---|---|---|
| `model/model.json` | `src/train.py` (oraz odtwarzany w `sprawdzenia/check_s3_s4.py`) | Zamrożony model `k-NN (k=3, L2)` z parametrami, cechami i SHA-256 `train.csv` |
| `wyniki/eksperymenty_validation.json` | `src/experiment.py` | Wyniki centroidu, 8 konfiguracji k-NN, 5 konfiguracji softmax, 5-fold CV i wyboru 7a–7c |
| `wyniki/validation_predictions.csv` | `src/predict.py` (tryb wsadowy na kopii validation bez `label`) | 200 przewidywań na `validation` (`image_id,image_path,predicted_label`) |
| `wyniki/validation_evaluation.json` | `src/evaluate.py` (`validation`) | Trafienia, pomyłki, macierz 10×10, mylone pary i błędne `image_id` na `validation` |
| `wyniki/test_predictions.csv` | `src/predict.py` (tryb wsadowy na kopii test bez `label`) | 200 przewidywań na `test` (`image_id,image_path,predicted_label`) |
| `wyniki/test_evaluation.json` | `src/evaluate.py` (`test`) | Trafienia, pomyłki, macierz 10×10, mylone pary i błędne `image_id` na `test` |
| `wyniki/s3_s4_sprawdzenia.json` | `sprawdzenia/check_s3_s4.py` | Raport odtwarzalności bajtowej S3 i inferencji w osobnym procesie S4 na `validation` (z odrzuceniem `label`) |
| `wyniki/s4_test.json` | `sprawdzenia/check_s4_test.py` | Sprawdzenie S4 na `test`: zgodność `test_predictions.csv` z modelem w pamięci i 3 pojedyncze obrazy testowe |
| `wyniki/s5_niezalezna_macierz.json` | `sprawdzenia/check_confusion_matrix.py` (`validation`) | Niezależna macierz 10×10 na `validation` (bez importów z `src/`) |
| `wyniki/s5_test_macierz.json` | `sprawdzenia/check_confusion_matrix.py` (`test`) | Niezależna macierz 10×10 na `test` (bez importów z `src/`) |
| `wyniki/sha_data_przed.json`, `wyniki/sha_data_po.json`, `wyniki/sha_data_po_tescie.json` | `sprawdzenia/check_data_sha.py` | Sumy SHA-256 wszystkich 1005 plików w `data/` przed pracą, po Kroku 2 i po teście |
| `wyniki/sha_przed_testem.json` | `sprawdzenia/freeze_before_test.py` | Zamrożone sumy SHA-256 `model/model.json` i 8 plików `src/` przed oceną testową |
| `wyniki/bramka_przed_testem.json`, `wyniki/sha_po_tescie_weryfikacja.json` | `sprawdzenia/verify_freeze.py` | Weryfikacja zgodności SHA-256 modelu i `src/` z `sha_przed_testem.json` (przed i po teście) |
| `wyniki/test_odniesienie_centroid.json` | `sprawdzenia/eval_centroid_test.py` | Punkt odniesienia: wynik najbliższego centroidu L2 na `test.csv` |
| `wyniki/mcnemar_test.json` | `sprawdzenia/mcnemar_test.py` | Dokładny dwustronny test McNemara (k-NN vs centroid na tych samych 200 obrazach testowych, bez importów z `src/`) |
| `wyniki/bledy_test.json` | `sprawdzenia/analyze_test_errors.py` | Szczegółowa analiza 5 błędów testowych (3 najbliżsi sąsiedzi z `train`, opisy, przedziały Wilsona) |
| `wyniki/bledy_test.png` | `sprawdzenia/analyze_test_errors.py` | Zbiorcza wizualizacja 5 błędnych obrazów testowych (×8 bez wygładzania) z 3 sąsiadami |
| `wyniki/softmax_zbieznosc_train.json` | `sprawdzenia/softmax_zbieznosc_train.py` | Badanie zbieżności regresji softmax na `train.csv` (4 kroki uczenia × 5 wartości `λ`) |
