# Laboratorium 4 — przegląd i odbiór rozwiązania z Labu 3

- Repozytorium: https://github.com/krzzbi/miob_lab (folder `lab4/`, oceniany projekt w `lab3/`)
- Aplikacja agenta: Google Antigravity · model: Gemini 4 Neon · poziom rozumowania: High

## 1. Cel i oceniany rezultat

Celem jest przegląd prototypu rozpoznawania spamu z Labu 3 i decyzja, czy przyjąć go w uzgodnionym zakresie. Wymagania biorę z [planu z Labu 2](../lab3/plan-lab2.md) i z [delegacji z Labu 3](../lab3/sprawozdanie.md) (sekcja „Delegacja”), razem ze zmianami uzgodnionymi w trakcie Labu 3: implementacja tylko w bibliotece standardowej, metoda Newtona zamiast L-BFGS, poprawione wzorce maskowania, dodane wzorce nagłówków RSS, zmieniona asercja testu `remove`.

Oceniany rezultat to [projekt](../lab3/project/):
- kod w `src/`,
- 20 testów w `tests/`,
- zapisane modele w `models/`,
- wyniki w `results/`.

Najważniejsze są trzy zachowania:
- uczenie i dobór ustawień tylko na danych uczących, ocena raz na `validation.csv`;
- przewidywanie z zapisanego modelu bez ponownego uczenia i bez kolumny `label`;
- porównanie przewidywań z etykietami tych samych wiadomości (po `message_id`).

## 2. Ustalenia code review

Agent przejrzał `src/`, `tests/`, `README.md` i `summary.md`, a potem wykonał 11 sprawdzeń. Skrypty są w [`sprawdzenia/`](sprawdzenia/), wyniki w [`wyniki/`](wyniki/). Ustalenia dzieliłem na trzy rodzaje:
- potwierdzony błąd: jest wejście, które daje zły wynik, albo wyraźna niezgodność z wymaganiem;
- ograniczenie: wada świadoma albo bez wpływu na wyniki;
- podejrzenie: coś, czego nie rozstrzygnięto.

Przy każdym ustaleniu zapisywałem też wpływ: czy zmienia wymagane zachowanie albo zapisane wyniki Labu 3.

**Potwierdzone błędy**

| # | Co | Skutek | Wpływ na wyniki Labu 3 |
|---|---|---|---|
| U1 | `<MONEY>` nie łapie kwoty z symbolem waluty po liczbie: „100$”, „100.000$”, „50 €” i „25£” dają `<NUM>` (w `preprocess.py` granica `\b` stoi po znaku `$`) | Kwota w takim zapisie nie wzmacnia sygnału spamu | 1/180 wiadomości uczących (spam), 0/60 w `validation`. Poprawka próbna na kopii (S10): ten sam model i C = 10, słownik 2632, CV bez zmian (58/60 spamów, 3/120 FP), 0 zmienionych decyzji na `validation`, największa zmiana oceny 6·10⁻⁵. Skutek pomijalny |
| U2 | Dosłowny tekst `<url>`, `<money>` itp. w treści wiadomości zamienia się w token specjalny | Treść może sztucznie dodać cechę | Brak (0 wystąpień w obu zbiorach) |
| U3 | Adres URL zakończony „!” połyka wykrzyknik, więc znika `<EXCLAM>` | Utrata części sygnału | Brak (0 wystąpień) |
| U5 | Komentarze i README twierdzą, że przewidywanie korzysta „wyłącznie z zapisanego pliku JSON” razem z regułami przetwarzania. W rzeczywistości reguły pochodzą z `preprocess.py`: zmiana wzorca w JSON nie zmieniła oceny (S6) | Dokumentacja wprowadza w błąd co do samowystarczalności modelu | Brak, bo dziś wzorce w JSON i w kodzie są identyczne (oba modele) |
| U7 | [`summary.md`](../lab3/project/results/summary.md) twierdzi, że usunięcie śladów źródła „nie pogorszyło jakości” | Nadinterpretacja. Ślady źródła ma 27/40 ham w `validation` (w `train` 85/120), a model bez maskowania opiera 9 z 60 najsilniejszych wag na takich śladach (`date`, `url`, `yahoo`, `groups`, `supplied`…). Oba warianty mają więc ten sam skrót, a równy wynik nic nie mówi o poczcie bez tych śladów. To samo zdanie w moim sprawozdaniu z Labu 3 było już zastrzeżone poprawnie | Brak na liczbach. Błąd dotyczy wniosku |

**Ograniczenia (bez wpływu na wyniki)**
- U4, U8: testy wymagań (b) i (a) z Labu 3 sprawdzały co innego, niż deklarowały. Test (b) liczy słownik we własnej pętli, a nie w kodzie produkcyjnym CV. Test (a) porównuje dwa odczyty tego samego JSON-a. Kod był poprawny (potwierdziły to S5 i S9), ale te testy nie wykryłyby regresji.
- U6: JSON nie zawiera reguł tokenizacji, a zapisanych w nim wzorców maskowania kod nie używa. Przyszła zmiana `preprocess.py` po cichu zmieni przewidywania ze starego pliku modelu.
- U9–U11 (drobne):
  - objaśnienie błędu przycina listę do 50 słów (na naszym FN bez skutku);
  - ocena nie sprawdza zgodności etykiet z progiem (S4: 0/60 niezgodności);
  - brak zapisu normy gradientu w JSON (S7 przeliczył ją niezależnie: 8·10⁻¹³);
  - perturbacje w diagnostyce Labu 3 tylko dopisywały stopkę do spamu.

**Wycofane podejrzenie.** S8 wskazało 2 „pozostałości stopek” w ham z `validation`. S11 pokazało, że to zwykła treść: adres `…@egroups.com` w artykule i temat „public mailing list sign up package”. Ustalenie wycofałem.

**Nierozstrzygnięte podejrzenia (niesprawdzane, bo nie dotyczą wymagań (a)–(f)):**
- czy kwoty pisane wielkimi literami („500 DOLLARS”) gubią `<CAPS>`;
- zachowanie solvera i PR-AUC w przypadkach brzegowych (dane rozdzielne liniowo, remisy ocen);
- odtwarzalność w innej wersji Pythona;
- wrażliwość na usunięcie śladów źródła z ham.

## 3. Zgodność z celem i dowody

| Wymagane zachowanie | Wynik sprawdzenia | Co potwierdza | Czego nie obejmuje |
|---|---|---|---|
| Testy z Labu 3 przechodzą na obecnym kodzie | 20/20 OK, 34 s ([s1_tests.json](wyniki/s1_tests.json)) | Kod się nie zepsuł od Labu 3 | Tylko to, co testy sprawdzają (ich luki opisuję w sekcji 2) |
| (e) Zapisane modele i wyniki odpowiadają obecnemu kodowi, a dane się nie zmieniły | Ponowny pełny przebieg na kopii danych: 9/9 plików identycznych bajtowo z `lab3/project/models/` i `results/`, 2/2 modele zgodne z [frozen_models_sha256.json](../lab3/project/results/frozen_models_sha256.json), kopia danych 2/2 zgodna z `manifest.json` ([s2_control_run.json](wyniki/s2_control_run.json)) | Zamrożone pliki da się odtworzyć z obecnego kodu i danych, więc żadna późniejsza zmiana kodu nie unieważniła wyników | Poprawności samego kodu: powtarzalny przebieg odtwarza też ewentualny błąd. Odtworzenie sprawdzone tylko na tym samym komputerze i w Pythonie 3.12.1 |
| (a) Przewidywanie z zapisanego modelu w osobnym procesie, bez uczenia i bez `label` | Test z Labu 3 porównuje osobny proces z tym samym plikiem JSON wczytanym w pamięci. W przeglądzie dodałem S9: oceny modeli z pamięci zaraz po uczeniu (przed zapisem) wobec `predict.py` w osobnym procesie na pliku bez kolumny `label`. Różnica 0,0; 0/60 różnic etykiet dla obu modeli; `--stdin` też 0,0 ([s9_in_memory_vs_subprocess.json](wyniki/s9_in_memory_vs_subprocess.json)) | Zapis do JSON i ponowny odczyt nie zmieniają ocen, a przewidywanie nie potrzebuje uczenia ani etykiet | Niezależności od kodu: JSON nie zawiera reguł tokenizacji, a reguły maskowania są w nim tylko opisem (patrz U5, U6). Zmiana `preprocess.py` zmieniłaby przewidywania z tego samego pliku |
| (b) Słownik i IDF tylko z części uczącej każdego foldu | Test z Labu 3 sprawdza własną pętlę, a nie kod produkcyjny. W przeglądzie dodałem S5: słowo-strażnik dopisane do 3 wiadomości z części walidacyjnej foldu 1 nie trafia do słownika tego foldu (rozmiar słownika bez zmian, oceny OOF różnica 0,0), a dopisane do części uczącej trafia (+1) ([s5_cv_isolation.json](wyniki/s5_cv_isolation.json)) | Kod CV używany do wyboru modelu i progu nie przenosi słów z części walidacyjnej do słownika | Wzorców maskowania: ustalono je ręcznie na całym `train.csv` przed CV, więc foldy nie są od nich niezależne |
| (c) Maskowanie usuwa artefakty i nie zmienia `remov…`/`unsubscrib…` w spamie | Na `train` testy z Labu 3. W przeglądzie S8 na `validation` (tylko diagnostycznie): spam z `remov` 10 → 10, z `unsubscrib` 3 → 3; wzorce RSS trafiły 0 spam / 8 ham; zostały 2 pozostałości stopek w ham ([s8_masking_and_tokens.json](wyniki/s8_masking_and_tokens.json)) | Maskowanie nie zabrało spamowi słów, które plan kazał zostawić, także na danych, na których wzorców nie ustalano | Pełnego usunięcia śladów źródła na nowych danych (2 pozostałości) ani `remove` w adresach URL (znika w `<URL>`, uzgodnione w Labie 3) |
| (d) Próg: największa czułość przy ≤ 3 FP | Test z Labu 3 z niezależną pętlą po ocenach OOF (`test_cv_pipeline.py`) przechodzi w S1. W przeglądzie S4: 0/60 wierszy, w których etykieta nie zgadza się z „ocena ≥ próg z JSON”, dla obu modeli ([s4_threshold_consistency.json](wyniki/s4_threshold_consistency.json)) | Test potwierdza, że próg 0,329 spełnia regułę na ocenach OOF; S4 potwierdza, że zapisane etykiety wynikają z tego progu | Reguły środka przedziału na samych ocenach OOF (test sprawdza ją tylko na małym przykładzie) ani przeniesienia progu z modeli uczonych na 144 wiadomościach na model uczony na 180 |
| (f) Porównanie przewidywań z etykietami tych samych wiadomości | S3: osobny skrypt bez kodu projektu łączy `validation_predictions.csv` z `label` po `message_id`: 60 par, 0 braków, 0 duplikatów, TP/FP/FN/TN = 19/0/1/40, miary zgodne z [validation_metrics.json](../lab3/project/results/validation_metrics.json) (różnica 0,0); to samo dla Naiwnego Bayesa ([s3_independent_cm.json](wyniki/s3_independent_cm.json)) | Liczby w wynikach Labu 3 dotyczą właściwych par przewidywanie–etykieta | Poprawności samych ocen (to S9 i S2) |
| Model końcowy jest dopasowany do minimum (‖grad‖ < 1e-6) | S7: gradient liczony własnym wzorem na całym `train`: 8,1·10⁻¹³; funkcja projektu daje to samo (różnica 6,6·10⁻¹⁶) ([s7_final_gradient.json](wyniki/s7_final_gradient.json)) | Optymalizacja zbiegła dla zapisanych wag | Przetwarzania tekstu (pochodzi z kodu projektu) i Naiwnego Bayesa |
| Przegląd niczego nie zmienił w `lab3/` | Sumy SHA-256 całego `lab3/` (39 plików + 16 w `__pycache__`) przed sprawdzeniami i po wszystkich sprawdzeniach: bez zmian ([sha_przed.json](wyniki/sha_przed.json), [sha_po.json](wyniki/sha_po.json)) | Ocena dotyczy dokładnie tego stanu, który był oddany w Labie 3 | — |

**Kluczowy dowód: przewidywanie z zapisanego modelu (S9).** Test z Labu 3 nie wystarczał. Porównywał wynik `predict.py` z tym samym plikiem JSON wczytanym drugi raz w pamięci. Gdyby zapis do JSON coś gubił (np. zaokrąglał wagi albo pomijał część słownika), oba odczyty byłyby tak samo błędne i test by przeszedł.

S9 porównuje z czymś spoza pliku: z modelem w pamięci zaraz po uczeniu, zanim został zapisany. Przewidywanie działa w osobnym procesie, który nie ma dostępu do obiektu z uczenia, i na kopii `validation.csv` bez kolumny `label`, więc nie może ani douczyć modelu, ani zajrzeć do etykiet. Różnica ocen 0,0 dla 60 wiadomości i obu modeli oznacza, że zapisany plik odtwarza model w całości.

Ten dowód wystarcza dla wymagania „przewidywanie bez uczenia i bez etykiet”. Nie wystarcza dla stwierdzenia, że sam plik JSON jest modelem. Przewidywanie zależy też od `preprocess.py` (U6), więc model trzeba przenosić razem z kodem w tej samej wersji.

## 4. Decyzja z uzasadnieniem

**Decyzja: przyjmuję rozwiązanie z Labu 3 w uzgodnionym zakresie, bez poprawek w ocenianym projekcie.**

Podstawa:
- Każde z wymaganych zachowań ma dowód, który nie polega na sprawdzanym kodzie:
  - (a) S9,
  - (b) S5,
  - (c) testy i S8,
  - (d) test z niezależną pętlą i S4,
  - (e) S2,
  - (f) S3.
- Zamrożone modele i wyniki da się odtworzyć bajt w bajt z obecnego kodu (S2), więc ocena dotyczy dokładnie tego, co oddałem w Labie 3.
- Jedyny potwierdzony błąd z wpływem na zapisany model (U1) dotyczy 1 ze 180 wiadomości uczących. Poprawka próbna nie zmieniła modelu, C, wyniku CV ani żadnej z 60 decyzji na `validation` (S10).
- Pozostałe potwierdzone błędy mają 0 wystąpień w danych (U2, U3) albo dotyczą dokumentacji (U5, U7).

Przed decyzją poprosiłem agenta o najmocniejsze argumenty przeciw. Żaden jej nie zmienił. Agent poprawił za to jedną rzecz w moim uzasadnieniu: dla (d) dowodem jest test z niezależną pętlą, a S4 sprawdza tylko zgodność etykiet z progiem.

**Do następnej wersji (nie teraz):**
- U1–U3 zmieniają reprezentację, czyli model. `validation` jest już zużyte, więc taką wersję trzeba ocenić na nowych danych.
- U5 i U7 (dokumentacja), U4 i U8 (testy, które mają wywoływać kod produkcyjny, jak S5 i S9) oraz U6 (zapis reguł tokenizacji w JSON albo sprawdzanie wersji kodu) nie zmieniają przewidywań i nie wymagają nowych danych. Zostawiam je na następną wersję tylko po to, żeby oceniany projekt z Labu 3 został w stanie z oddania.

**Ograniczenia, które zostają:**
- `validation` ma tylko 60 wiadomości (1 spam = 5 pp, 1 ham = 2,5 pp) i pochodzi z tych samych archiwów co `train`. Wynik 19/20 przy 0/40 fałszywych alarmów nie mówi, jak model działa na innej poczcie.
- Margines jest minimalny: przeoczony spam ma ocenę 0,217, najwyżej oceniony ham 0,200.
- Próg 0,329 wybrano na ocenach OOF z modeli uczonych na 144 wiadomościach (słowniki 2010–2268 słów). Własne progi foldów przy 0 FP leżą w przedziale 0,347–0,530. Nie wiadomo, czy model uczony na 180 wiadomościach ma tę samą skalę ocen.
- Wśród 30 najsilniejszych słów ham są słowa z tematów list technicznych i z wątków (`re`, `wrote` 36 ham / 0 spam, `rpm` 8/0, `linux` 9/0, `exmh`, `cvs`, `david`). Wśród słów spamu są fragmenty HTML i krótkie tokeny o niejasnym pochodzeniu (`br`, `ne` 0/2, `su` 1/1). Przypuszczam, że model uczy się częściowo tematyki korpusu z 2002 r., a nie samych cech spamu. Nie sprawdzałem tego.
- Wszystkie przebiegi wykonano w Pythonie 3.12.1 na jednym komputerze.

**Nowe wyniki z przeglądu:**
- skrypty: [`lab4/sprawdzenia/`](sprawdzenia/);
- wyniki: [`lab4/wyniki/`](wyniki/):
  - `s1_tests.json`: testy,
  - `s2_control_run.json`: odtworzenie wyników,
  - `s3_independent_cm.json`: macierz pomyłek,
  - `s4_threshold_consistency.json`: zgodność etykiet z progiem,
  - `s5_cv_isolation.json`: izolacja słownika w CV,
  - `s6_json_self_sufficiency.json`: samowystarczalność JSON-a,
  - `s7_final_gradient.json`: gradient modelu końcowego,
  - `s8_masking_and_tokens.json`: maskowanie i tokeny,
  - `s9_in_memory_vs_subprocess.json`: przewidywanie w osobnym procesie,
  - `s10_money_fix_effect.json`: skutek poprawki próbnej U1,
  - `s11_masking_claim.json`: twierdzenie o maskowaniu,
  - `sha_przed.json` i `sha_po.json`: identyczne, czyli przegląd nie zmienił niczego w `lab3/`.
