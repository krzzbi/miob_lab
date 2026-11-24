# Laboratorium 5 — rozpoznawanie odręcznie zapisanych cyfr

Repozytorium: https://github.com/krzzbi/miob_lab (folder `lab5/`)
Aplikacja agenta: Google Antigravity · model: Gemini 4 Neon · poziom rozumowania: High

## 1. Podejście, decyzje i rozważana alternatywa

- **Środowisko:** tylko biblioteka standardowa Pythona 3.12, bez numpy i Pillow. Format danych na to pozwala: wszystkie 1000 PNG to 8×8, 8 bit, szarość, filtr 0 i 17 poziomów jasności (`round(k·255/16)`). Dekoder obsługuje filtry 0–4, a każdy inny format odrzuca.
- **Kandydaci i punkt odniesienia:** k-NN (k ∈ {1, 3, 5, 7} × L1/L2) oraz regresja softmax z L2 (λ ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}). Punktem odniesienia jest najbliższy centroid. MLP odrzuciłem, bo wynik zależy od ziarna, a przy 600 przykładach łatwo o przeuczenie.
- **Kryteria wyboru (ustalone przed uczeniem):** „wyraźna przewaga” to ≥ 3/200 trafień więcej na validation i nie gorsza CV. Bez wyraźnej przewagi decyduje 5-krotna CV na train, a przy różnicy < 0,5 pp wygrywa k-NN.
- **Wybór** ([eksperymenty_validation.json](wyniki/eksperymenty_validation.json)):
  - najlepszy k-NN to k = 3, L2: 193/200 na validation, CV 578/600 (k = 1 ma też 193/200, ale niższą CV, 573/600);
  - najlepszy softmax to λ = 0: 191/200, CV 566/600;
  - różnica 2/200 nie jest wyraźną przewagą, więc rozstrzyga CV: 12/600 (2 pp) na korzyść k-NN;
  - centroid ma 178/200, więc oba modele są wyraźnie lepsze.
- **Za k-NN wobec softmaxu przemawiało:** wyższa CV, brak iteracyjnej optymalizacji oraz powtarzalność. Softmax z λ = 0 zatrzymał się na limicie 1500 epok, więc jego wynik zależy od arbitralnie ustawionego limitu. Kosztem jest model, który przechowuje cały zbiór uczący (761 kB).
- **Model końcowy** uczony tylko na train (600), bez douczania na validation. Inaczej przed testem nikt nie sprawdziłby modelu, który oddaję.

## 2. Współpraca z agentem

Pracowałem krokami, a każdy kończył się raportem i punktem zatrzymania:
1. tylko odczyt danych i środowiska;
2. główne zlecenie: uczenie i wybór na train i validation, zakończone zamrożeniem modelu;
3. jednorazowa ocena na teście, na którą dałem osobną zgodę.

Po każdym raporcie sam przeliczałem kluczowe liczby własnym dekoderem PNG i własnym k-NN, bez kodu agenta: wszystkie 9 wyników siatki na validation, 195/200 i 186/200 na teście, 0/200 różnic z plikami przewidywań. Agent nie znał tych liczb z góry.

### Przekazane zlecenie i uzasadnienie jego zakresu

Poniżej główne zlecenie (Krok 2) w takiej postaci, w jakiej je przekazałem. Agent przed pisaniem kodu wskazał w nim sprzeczność: SHA całego `data/` wymaga odczytu bajtów `test.csv`, mimo zakazu jego używania. Rozstrzygnąłem ją krótką odpowiedzią: wolno liczyć tylko bajtowe SHA, bez parsowania. Doprecyzowałem też remisy odległości w k-NN i konwencję kary L2.

<details><summary>Treść zlecenia (Krok 2)</summary>

```text
Krok 2 — główne zlecenie. Zasady z Kroku 1 obowiązują dalej, z jedną zmianą: README danych przeniosłem do lab5/data/README.md (chroniony jak cała lab5/data/). Nowy lab5/README.md utworzysz dopiero w późniejszym kroku, na moje polecenie. Odpowiedzi na Twoje pytania są wbudowane poniżej.

## Rezultat
Klasyfikator cyfr w czystym Pythonie 3.12 (tylko biblioteka standardowa, bez instalacji), uczony wyłącznie na train.csv, z ustawieniami wybranymi na validation.csv, zapisany w lab5/model/ i zamrożony (SHA-256) przed oceną na teście. Skrypt przewidywania przyjmuje ścieżkę do PNG i zwraca cyfrę bez etykiety i bez ponownego uczenia.

## Kontekst
- train i validation pochodzą z optdigits.tra, test z optdigits.tes (inni autorzy pisma), więc wynik na validation może być optymistyczny.
- Test ma być niezależną, jednorazową oceną, dlatego w tym kroku nie jest używany w ogóle.

## Zakres
1. Dekoder PNG (zlib + struct): 8×8, 8 bitów, typ koloru 0, bez przeplotu, filtry wierszy 0–4, sprawdzanie CRC. Każdy inny format odrzucany z czytelnym komunikatem, bez zgadywania.
2. Cechy: 64 piksele podzielone przez 255. Każde inne przekształcenie tylko po opisaniu go i dopasowaniu wyłącznie na train.
3. Punkt odniesienia (nie kandydat): najbliższy centroid klasy, odległość L2.
4. Kandydat A, k-NN: k ∈ {1, 3, 5, 7} × metryka {L2, L1}, głosy równe; remis głosów rozstrzyga najbliższy sąsiad spośród remisujących klas (8 konfiguracji).
5. Kandydat B, regresja softmax z regularyzacją L2: inicjalizacja zerami, pełny gradient na całym train, λ ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1} (5 konfiguracji). Krok uczenia i liczbę epok ustal na train na podstawie zbieżności straty i uzasadnij w raporcie.
6. Dla każdej konfiguracji: trafienia na validation (x/200). Dla konfiguracji remisujących w kryteriach poniżej oraz dla najlepszej konfiguracji A, najlepszej B i punktu odniesienia: 5-krotna CV na train (foldy warstwowe po klasach, stałe ziarno), średnia i zakres trafności foldów.
7. Wybór według kryteriów ustalonych z góry:
   a) w obrębie kandydata: najwięcej trafień na validation; remis → wyższa średnia CV na train; dalej remis → większe k / większe λ;
   b) między A i B: wyraźna przewaga = co najmniej 3 trafienia więcej na validation i nie gorsza średnia CV na train; bez wyraźnej przewagi → wyższa średnia CV na train; różnica CV poniżej 0,5 pp → k-NN;
   c) jeśli wybrany kandydat nie ma więcej trafień na validation niż punkt odniesienia, zatrzymaj się i zgłoś.
8. Model końcowy: wybrana konfiguracja uczona na całym train (600). Bez douczania na validation.
9. Zapis modelu w JSON ze wszystkim, co potrzebne do użycia: typ modelu, wersja formatu, ustawienia, opis przekształcenia cech, parametry (dla k-NN wektory uczące i etykiety), SHA-256 pliku train.csv.
10. predict.py: (a) jedna ścieżka PNG → wypisuje cyfrę; (b) tryb wsadowy: CSV z kolumnami image_id,image_path (bez label) → CSV image_id,image_path,predicted_label. Skrypt nie wczytuje żadnego pliku z etykietami.
11. evaluate.py: łączy plik przewidywań z CSV z etykietami po image_id; liczy trafienia, pomyłki, macierz pomyłek 10×10, najczęściej mylone pary (prawdziwa → przewidziana) i listę błędnych obrazów z image_id. W tym kroku uruchamiany tylko dla validation.

## Zasada danych
- test.csv i obrazy testowe: w tym kroku nie otwieraj, nie wczytuj, nie przewiduj i nie licz na nich niczego.
- Wszystkie decyzje zapadają na train (CV) i validation.
- lab5/data/ tylko do odczytu: SHA-256 wszystkich plików lab5/data/ przed pracą (lab5/wyniki/sha_data_przed.json) i po pracy (sha_data_po.json); muszą być identyczne.
- Python uruchamiaj z -B i PYTHONDONTWRITEBYTECODE=1.

## Sprawdzenia
Oczekiwany wynik ma pochodzić spoza sprawdzanego kodu i być zapisany przed uruchomieniem. Nie zmieniaj oczekiwanego wyniku, żeby sprawdzenie przeszło — to wymaga konsultacji. Przy wyniku niezgodnym z oczekiwanym zatrzymaj się i nie wykonuj sprawdzeń od niego zależnych. Wyniki mają dotyczyć oddawanego kodu i zapisanego modelu: po każdej zmianie kodu powtórz sprawdzenia, których ona dotyczy.
- S1 Dekoder, przykład ręczny: data/images/digit-17a28342e8c0.png ma wiersz 0 = [0, 0, 128, 191, 191, 0, 0, 0] i wiersz 7 = [0, 0, 143, 175, 191, 191, 64, 0] (odczytane niezależnie, innym narzędziem).
- S2 Dekoder, filtry i odrzucanie: syntetyczne PNG z zadanej macierzy 8×8 zakodowane filtrami 1–4 przez koder napisany w teście (nie przez kod z src/); oczekiwany wynik = zadana macierz. Odrzucenie: RGB, 16 bitów, przeplot, rozmiar inny niż 8×8, błędne CRC.
- S3 Odtwarzalność: dwa niezależne uczenia modelu końcowego dają pliki modelu identyczne bajtowo.
- S4 Przewidywanie w osobnym procesie: predict.py w trybie wsadowym na kopii validation bez kolumny label (w lab5/proby/) daje przewidywania identyczne z przewidywaniami modelu z pamięci przed zapisem (liczba różnic na 200). Tryb pojedynczej ścieżki dla 3 obrazów zgodny z trybem wsadowym.
- S5 Niezależna macierz pomyłek: skrypt w lab5/sprawdzenia/ bez importu z src/ liczy macierz z pliku przewidywań validation i validation.csv; musi być identyczna z wynikiem evaluate.py.
- S6 Punkt odniesienia: wynik najbliższego centroidu na validation i w CV.
- S7 SHA-256 danych przed i po.
- S8 Zamrożenie: SHA-256 pliku modelu i wszystkich plików src/ w lab5/wyniki/sha_przed_testem.json.
Testy jednostkowe w lab5/tests/ (unittest), uruchamiane jednym poleceniem; podaj wynik x/y i czas.

## Samodzielność
Sam decydujesz o podziale kodu w src/, nazwach, formacie JSON oraz kroku uczenia i liczbie epok softmax (z uzasadnieniem). Błędy we własnym kodzie poprawiasz sam i opisujesz je w raporcie. Zatrzymaj się i zapytaj, gdy: potrzebna jest instalacja; trzeba by zmienić dane, etykiety lub podział; wynik sprawdzenia jest niezgodny z oczekiwanym; zachodzi przypadek 7c; chcesz zmienić siatkę lub kryteria wyboru; cokolwiek wymagałoby użycia test.csv.

## Punkt zatrzymania
Po S8 zatrzymaj się. Nie uruchamiaj niczego na test.csv. Na ocenę testu dam osobną zgodę.

## Zakończenie — raport
- tabela: model | ustawienia | trafienia validation x/200 | CV średnia i zakres (gdzie liczona);
- zastosowanie kryteriów 7a–7c krok po kroku, z liczbami; wybrany model;
- validation dla wybranego modelu: trafienia, pomyłki, macierz 10×10, najczęściej mylone pary, lista błędnych image_id;
- tabela sprawdzeń: sprawdzenie | oczekiwany wynik (skąd) | wynik | czego nie obejmuje;
- lista: który skrypt co zapisuje lub nadpisuje;
- polecenia odtworzenia uczenia, przewidywania i oceny;
- czasy uczenia i przewidywania.

## Pliki
lab5/src/, lab5/tests/, lab5/sprawdzenia/, lab5/model/, lab5/wyniki/; brudnopis tylko w lab5/proby/. Nie twórz README.md ani sprawozdanie.md.

Zanim napiszesz kod, wskaż sprzeczności lub niejasności w tym zleceniu.
```
</details>

**Dlaczego taki zakres:**
- jedno zlecenie obejmuje wszystko, co wolno zrobić bez testu: dekoder, uczenie, wybór modelu, zapis, przewidywanie i sprawdzenia;
- koniec zlecenia wyznacza zamrożenie modelu (S8);
- siatkę i kryteria wyboru ustaliłem z góry, łącznie z progiem „wyraźnej przewagi” i regułą remisu, żeby wyboru nie dało się dopasować do wyników;
- oczekiwane wyniki sprawdzeń pochodzą spoza sprawdzanego kodu: wartości pikseli odczytane innym narzędziem, macierz liczona skryptem bez importu `src/`, przewidywanie w osobnym procesie z pliku bez etykiet.

### Moment, w którym wynik sprawdzenia zmienił działanie

Agent wybierał krok uczenia softmaxu, badając zbieżność na samym train ([softmax_zbieznosc_train.json](wyniki/softmax_zbieznosc_train.json)):
- przy `lr = 2.0` i λ = 0,1 uczenie nie zbiegło się: po 1500 epokach strata 1,99 i tylko 326/600 trafień na train;
- przy `lr = 0.5` λ = 1e-3 nie osiągnęło progu w 1500 epokach;
- dlatego przyjął `lr = 1.0`, czyli połowę wartości, która oscyluje.

To badanie początkowo leżało w brudnopisie poza repo i wypisywało wyniki tylko na ekran. Po moim przeglądzie agent przeniósł je do `sprawdzenia/` i zapisał wynik do pliku. Ponowne uruchomienie dało 20/20 identycznych przebiegów.

## 3. Wyniki oceny i omówienie pomyłek

Model k-NN (k = 3, L2) oceniłem raz, po zamrożeniu: SHA modelu i `src/` są zgodne przed testem i po nim ([bramka_przed_testem.json](wyniki/bramka_przed_testem.json), [sha_po_tescie_weryfikacja.json](wyniki/sha_po_tescie_weryfikacja.json)).

| Zbiór | k-NN (k = 3, L2) | 95% przedział Wilsona | Centroid (odniesienie) |
|---|---:|---|---:|
| validation (200) | 193/200 | 92,9–98,3% | 178/200 |
| **test (200)** | **195/200** | **94,3–98,9%** | 186/200 |

Pliki: [test_predictions.csv](wyniki/test_predictions.csv), [test_evaluation.json](wyniki/test_evaluation.json) (macierz 10×10), [s5_test_macierz.json](wyniki/s5_test_macierz.json) (niezależna macierz, zgodna), [test_odniesienie_centroid.json](wyniki/test_odniesienie_centroid.json).

**Przewaga nad centroidem** na tych samych obrazach: 10 obrazów rozpoznał dobrze tylko k-NN, 1 tylko centroid. Dokładny test McNemara daje p ≈ 0,012 ([mcnemar_test.json](wyniki/mcnemar_test.json)). Przewidywania w pliku zgadzają się z modelem w pamięci (0/200 różnic), a tryb jednej ścieżki z plikiem (3/3) ([s4_test.json](wyniki/s4_test.json)).

**Według kryteriów ustalonych przed testem** (≥ 183/200 i lepiej niż centroid) wynik mieści się w oczekiwaniach. Test jest o 2 trafienia lepszy od validation, choć pochodzi od innych autorów pisma. Centroid też poprawił się o 8 trafień. Przedziały się pokrywają, więc nie wnioskuję z tego, że model „działa lepiej na nowych autorach”. Przypuszczam raczej, że wylosowane 200 obrazów testowych było po prostu łatwiejsze.

**Najczęściej mylone cyfry:**
- test: każda z 5 pomyłek to inna para (3→8, 7→9, 9→4, 4→1, 1→6);
- validation: powtarzają się 5→9 (2) i 8→1 (2).

Przy 5–7 błędach na 200 nie ma pary, która wyraźnie dominuje.

**Omówienie błędów na teście** ([bledy_test.png](wyniki/bledy_test.png): obraz testowy i 3 najbliżsi sąsiedzi z train; [bledy_test.json](wyniki/bledy_test.json)):
- **Przegłosowany dobry sąsiad (3 z 5):** 3→8, 4→1 i 1→6. Najbliższy sąsiad ma dobrą etykietę, ale przegłosowują go dwaj dalsi. Najwyraźniej widać to przy `dbf4f7702a6f` (3→8): poprawna „3” leży w odległości 0,76, a dwie „8” w 1,56 i 1,65. Głosy równe niezależnie od odległości są tu słabością tego modelu. Na validation tak było w 2 z 7 błędów.
- **Brak bliskiego wzorca (1 z 5):** `09e2a05d6bd2` (7→9). Wszyscy trzej sąsiedzi to „9”, i to najdalsi spośród wszystkich błędów (1,82–1,88). Na powiększeniu ta „7” ma zamkniętą pętlę u góry i przypomina „9”. W train brakuje podobnie zapisanej „7”.
- **Kształt pośredni (1 z 5):** `7043c36778f2` (9→4). Pętla jest otwarta u góry, więc obraz wygląda jak „4”. Dwaj najbliżsi sąsiedzi to „4”.

## 4. Czy przyjąć prototyp

**Decyzja: przyjmuję, w zawężonym zakresie.** Zakres obejmuje przygotowane, wycięte obrazy pojedynczych cyfr 8×8 w tej samej postaci co dane UCI optdigits: ciemne tło, jasny znak, 17 poziomów jasności i ten sam sposób wycinania. Prototyp nie nadaje się do bezobsługowego odczytu pól formularzy.

**Podstawy przyjęcia:**
- **Wynik:** 195/200 na teście (95% Wilson 94,3–98,9%), lepiej niż centroid (McNemar p ≈ 0,012) i powyżej progu 183/200 ustalonego przed testem.
- **Brak dopasowania do testu:** wybór modelu zamknąłem przed testem. SHA modelu i `src/` jest takie samo przed testem i po nim.
- **Użycie zgodne z wymaganiem:** przewidywanie idzie z zapisanego modelu, w osobnym procesie, z pliku bez etykiet (0/200 różnic na validation i na teście).
- **Odtwarzalność:** odtworzenie od zera według README dało model i przewidywania identyczne bajtowo.

**Istotne ograniczenia:**
- **Mała próba:** test to 200 obrazów, po 20 na klasę. Dla klasy z wynikiem 19/20 dolna granica przedziału to ok. 76%, więc o skuteczności pojedynczych cyfr wiadomo niewiele.
- **Pola wielocyfrowe:** przy błędzie 2,5–5,7% na cyfrę pole 6-cyfrowe miałoby błąd w ok. 14–30% przypadków (przypuszczenie, przy założeniu niezależnych błędów). Odczyt formularzy wymaga więc weryfikacji ręcznej.
- **Brak odrzucania:** model zawsze zwraca cyfrę, także dla pustej kratki czy szumu, i nie podaje pewności. W 3 z 5 błędów testowych dobry najbliższy sąsiad przegrał głosowanie 1:2.
- **Autorzy pisma:** to, że test pochodzi od innych autorów, wiem z opisu danych, nie z samych danych. Pliki nie mają identyfikatorów autorów, więc nie da się rozbić wyniku na autorów.
- **Wybór między modelami:** k-NN i softmax dzieliły na validation tylko 2/200. Rozstrzygnęła jedna 5-krotna CV (jedno ziarno), a softmax z λ = 0 zatrzymał limit epok. Nie twierdzę więc, że k-NN jest ogólnie lepszy od modelu liniowego.
- **Środowisko:** wszystko sprawdziłem w jednym środowisku (macOS, Python 3.12.1). Model ma 761 kB, a czas przewidywania rośnie liniowo z liczbą przykładów uczących (ok. 0,05 s na obraz z wiersza poleceń).

**Niewykonane sprawdzenia:**
- dane z docelowych formularzy (inne skanowanie, ciemny tusz na jasnym tle);
- odporność na przesunięcia o 1–2 piksele, obroty, szum, puste obrazy i skreślenia;
- inne systemy i wersje Pythona;
- dekoder PNG sprawdzony zewnętrznym narzędziem (np. Pillow) na wszystkich 1000 plikach. W repo jest tylko S1 (2 wiersze 1 pliku) i S2 (syntetyczne PNG). Moja kontrola własnym dekoderem spoza repo objęła 412 obrazów (wszystkie z validation i testu oraz 12 z train) i dała identyczne przewidywania, ale nie jest oddanym dowodem;
- powtarzana CV z wieloma ziarnami;
- analiza obrazkowa błędów na validation.

**Pozostała praca (następna wersja, oceniana na nowych danych, bo test został już użyty):**
1. Głosy ważone odległością albo próg pewności oparty na odległości do najbliższego sąsiada (za: 3/5 błędów testowych i 2/7 na validation miały dobrego najbliższego sąsiada).
2. Odrzucanie niepewnych wejść do ręcznej weryfikacji.
3. Zebranie danych z docelowych formularzy i test odporności na przesunięcia.
