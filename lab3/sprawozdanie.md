# Laboratorium 3 — od planu do działającej implementacji

Repozytorium: https://github.com/krzzbi/miob_lab (folder `lab3/`)
Agent: Antigravity · Gemini 4 Neon · poziom rozumowania: High

Punkt startowy: plan z Labu 2 (`plan-lab2.md`) i te same dane (train 180: 120 ham / 60 spam, validation 60: 40 / 20; sumy SHA-256 zgodne z `data/manifest.json`). Środowisko: Python 3.12.1 bez numpy, scipy i scikit-learn, ok. 11 GiB wolnego miejsca na dysku.

## 1. Delegacja

Główne zlecenie przekazane agentowi przed implementacją (po ustaleniu punktu startowego i podziału pracy):

> **Rezultat.** W `project/` ma powstać lokalny prototyp realizujący plan z `plan-lab2.md`: przygotowanie tekstu (maskowanie śladów źródła, tokeny `<CAPS>`, `<EXCLAM>`, `<URL>`, `<EMAIL>`, `<MONEY>`, `<NUM>`), TF-IDF pojedynczych słów z min_df=2, regresja logistyczna L2 (model główny) i wielomianowy Naiwny Bayes (alternatywa), wybór ustawień, modelu i progu 5-krotną walidacją krzyżową na train, uczenie końcowe na całym train, jednorazowa klasyfikacja i ocena validation z punktami odniesienia oraz diagnostyka z sekcji 5 planu. Model, słownik, IDF i próg zapisane w JSON tak, żeby przewidywać bez ponownego uczenia. README projektu opisuje zależności, uruchomienie i lokalizacje wyników.
>
> **Kontekst.** Dane: `project/data/train.csv` (180: 120 ham / 60 spam) i `validation.csv` (60: 40 / 20), sumy zgodne z `manifest.json`. Python: `/Users/jasminetea/.pyenv/shims/python3` (3.12.1), tylko biblioteka standardowa, bez instalacji pakietów. Doprecyzowania planu: maskowanie usuwa artefakty, a wzorce ustalasz tylko na train. Siatka: C ∈ {0.1, 1, 10, 100}, α ∈ {0.01, 0.1, 1}. Ziarno 2026. Remisy: wyższe PR-AUC, dla progu mniej FP, potem środek przedziału. Reguła odniesienia „`<MONEY>` i `remove` → spam” jest zamrożona.
>
> **Zakres.** Pięć części w kolejności, jak w Twojej tabeli: (1) dane, maskowanie, TF-IDF; (2) własne modele, miary, zapis JSON; (3) CV, wybór, uczenie końcowe, najsilniejsze wagi; (4) przewidywanie bez `label`, ocena validation, punkty odniesienia; (5) diagnostyka po ocenie, pełny przebieg, README. Poza zakresem: zmiany w `project/data/`, `plan-lab2.md`, `lab-3-assignment.md`, `sprawozdanie.md`, instalacja pakietów, sieć, skrzynka pocztowa, linki z wiadomości oraz strojenie czegokolwiek pod validation.
>
> **Zasada danych.** Wszystkie decyzje zapadają na train. Validation klasyfikujesz i oceniasz raz, dopiero po mojej zgodzie. Skrypt przewidujący nie czyta kolumny `label`. Diagnozy po ocenie niczego nie zmieniają w modelu.
>
> **Sprawdzenia.** Testy `unittest` w `project/tests/`, uruchamiane z `project/`. Oczekiwany wynik ma pochodzić z planu, z ręcznie policzonego małego przykładu albo z niezależnego zliczenia na danych, nigdy z wyniku sprawdzanego kodu. Wymagane zachowania:
> (a) przewidywanie z plików JSON w osobnym procesie na CSV bez kolumny `label` daje te same oceny co przebieg uczący (różnica < 1e-12);
> (b) słownik i IDF pochodzą tylko z części uczącej, w każdym przebiegu CV i w uczeniu końcowym;
> (c) maskowanie nie zmienia liczby wiadomości spam w train zawierających `remove` (porównanie z niezależnym zliczeniem przed maskowaniem) i usuwa wskazane artefakty;
> (d) wybrany próg daje ≤ 3 FP na 120 ham w ocenach z CV i największą czułość przy tym ograniczeniu (sprawdzenie niezależną pętlą);
> (e) dwa kolejne pełne przebiegi dają identyczne pliki modelu i przewidywań, a sumy SHA-256 plików w `data/` się nie zmieniają;
> (f) macierz pomyłek sumuje się do 60 = 40 + 20, a „zawsze ham” daje 40/60.
> Do tego dochodzą sprawdzenia własnych implementacji: gradient porównany z różnicami skończonymi, gradient ≈ 0 po dopasowaniu, Naiwny Bayes porównany z ręcznym przykładem. Liczba słów w słowniku (plan: ok. 2650–2900) jest kontrolą orientacyjną. Nie zatrzymuje pracy, ale wynik spoza 2000–3500 zgłaszasz.
>
> **Samodzielność.** Po każdym uruchomieniu sprawdzeń czytasz wynik. Błąd w kodzie poprawiasz sam i uruchamiasz sprawdzenia ponownie. Nie zmieniasz oczekiwanego wyniku testu po to, żeby test przeszedł: to wymaga konsultacji. Konsultujesz ze mną przed dalszą pracą, gdy trzeba odejść od planu lub powyższych decyzji, gdy wzorzec maskowania mógłby wyciąć treść spamu, gdy solver nie zbiega albo CV trwa dłużej niż ok. 30 s, gdy wybór między regresją a Naiwnym Bayesem jest graniczny, gdy najsilniejsze wagi pokazują ślady źródła oraz po trzech nieudanych próbach naprawy tego samego problemu. Punkty zatrzymania: po części 2 (raport), po części 3 (bramka: wyniki CV, wybrany model, ustawienia, próg, 30 najsilniejszych wag i lista wzorców maskowania; `validation.csv` otwierasz dopiero po moim „zgoda na ocenę”) oraz po części 5 (raport końcowy). Każdy raport zawiera: co powstało, które sprawdzenia uruchomiłeś i z jakim wynikiem (liczby), co poprawiłeś po nieudanym sprawdzeniu i jakie są odstępstwa od planu.
>
> **Zakończenie.** Zlecenie jest zakończone, gdy wszystkie testy przechodzą, polecenia z README odtwarzają cały przebieg w czystym procesie, w `models/` jest model z reprezentacją i progiem, w `results/` są wyniki CV, przewidywania dla validation (`message_id`, ocena, przewidywana etykieta), miary, lista błędów i diagnostyka, a `results/summary.md` podsumowuje przebieg z listą odstępstw i niewykonanych sprawdzeń.
>
> **Pliki.** Kod: `project/src/`, testy: `project/tests/`, model: `project/models/`, wyniki i podsumowanie: `project/results/`, próby: `project/proby/`, README: `project/README.md` (zastępuje README paczki z Labu 2).

Korekty zlecenia w trakcie pracy:
- Po części 2 dodałem do sprawdzeń maskowania trzy elementy: liczbę wiadomości w train dopasowanych przez każdy wzorzec, wyszukiwanie pozostałości stopek po maskowaniu oraz odmiany `remov…` i `unsubscrib…` w spamie (przed i po maskowaniu). Agent zmienił też jedną asercję testu bez konsultacji. Zaakceptowałem zmianę po fakcie, bo była merytorycznie poprawna, i przypomniałem o zasadzie.
- Na bramce przed validation rozszerzyłem maskowanie o nagłówki kanałów RSS (`URL: …`, `Date: Not supplied` i podobne linie `Date:`). Ustaliłem z góry kryterium dodania wzorca i to, co oznacza „wyraźna przewaga” Naiwnego Bayesa (opis w sekcji 3).

## 2. Uzasadnienie organizacji pracy

**Biblioteka standardowa zamiast scikit-learn.** W Pythonie 3.12.1 nie ma numpy, scipy ani scikit-learn. Instalacja w `.venv` to ok. 200 MB zależności przy 11 GiB wolnego miejsca. Model zapisany przez joblib byłby też związany z wersjami bibliotek. Wybraliśmy bibliotekę standardową i zapis w JSON, który da się przeczytać, co i tak jest potrzebne do przeglądu wag z sekcji 5 planu. Koszt: TF-IDF, regresję i Naiwnego Bayesa piszemy sami, więc ich poprawność trzeba sprawdzać osobno.

**Pięć części z trzema punktami zatrzymania zamiast jednego zlecenia.** Przy własnej implementacji drobny błąd (w gradiencie, w normalizacji TF-IDF, wyciek słownika w CV) dałby wiarygodnie wyglądające, ale błędne miary. W jednym dużym zleceniu wyszedłby dopiero na końcu albo wcale. Dlatego każda część kończy się sprawdzeniem, zanim kolejna z niej skorzysta. Najważniejszy punkt zatrzymania to bramka przed validation. Wynika z zasady planu: wszystkie decyzje zapadają na train, a validation liczy się raz. Jedyna dozwolona korekta maskowania (po przeglądzie wag modelu uczonego na train) musi więc nastąpić przed oceną. Odrzuciłem też zatrzymanie po każdej z pięciu części. Części 1 i 2 sprawdzają się testami na ręcznie policzonych przykładach, więc wspólny raport po części 2 wystarcza.

**Źródła feedbacku.** Oczekiwany wynik pochodzi spoza sprawdzanego kodu:
- ręcznie policzone małe przykłady (tokeny, TF-IDF, Naiwny Bayes, miary i próg);
- własności matematyczne (gradient zgodny z różnicami skończonymi, gradient ≈ 0 w minimum funkcji ściśle wypukłej);
- niezależne zliczenia na danych (liczba spamów z `remove` przed i po maskowaniu, liczności foldów 24/12);
- wymagania z treści zadania (przewidywanie po ponownym wczytaniu bez uczenia, CSV bez kolumny `label`);
- sumy SHA-256 z `manifest.json`.

Same testy składni lub stylu nie potwierdziłyby wymaganego działania, dlatego punkty (a)–(f) sprawdzają zachowanie rozwiązania.

## 3. Moment wykorzystania feedbacku

**Sprawdzenie:** dla każdego z 22 wzorców maskowania liczba wiadomości w train, które dopasowuje, oraz wyszukanie pozostałości stopek po maskowaniu. Oczekiwany wynik wynikał z tego, jak wzorce powstały: skoro ustalono je na train, każdy powinien coś dopasować, a pozostałości powinno być 0.

**Wynik:** wzorzec banera „Yahoo! Groups Sponsor” dopasował 0 wiadomości, a fraza „Groups Sponsor” została po maskowaniu w 7 wiadomościach ham. Podobnie ogólna stopka Mailmana: „listinfo” zostało w 2 wiadomościach. Wcześniejsze testy przechodziły, bo nie sprawdzały, czy każdy wzorzec cokolwiek dopasowuje.

**Dalsze działanie:** agent poprawił wyrażenie, bo baner w danych kończy się `~-->`, a wzorzec oczekiwał `~->`. Rozszerzył też ogólną stopkę i dodał stopkę jednej listy. Po poprawce wzorzec banera dopasowuje 7 ham i 0 spam, pozostałości spadły 7 → 0 i 2 → 0, a liczba spamów z `remov…` (33) i `unsubscrib…` (9) się nie zmieniła. Bez tej poprawki 7 wiadomości ham miałoby w reprezentacji słowa banera, czyli ślad źródła, którego plan kazał się pozbyć.

## 4. Rezultat i ograniczenia

**Linki:** [README projektu](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/README.md) · [model](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/models/spam_classifier.json) · [przewidywania dla validation](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/results/validation_predictions.csv) · [miary](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/results/validation_metrics.json) · [błędy](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/results/validation_errors.json) · [wyniki CV](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/results/cv_results.json) · [diagnostyka](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/results/diagnostics_section5.json) · [podsumowanie agenta](https://github.com/krzzbi/miob_lab/blob/main/lab3/project/results/summary.md)

**Rezultat.** Zgodnie z README cały przebieg uruchamia jedno polecenie. Obejmuje ono weryfikację danych, CV i uczenie, przewidywanie w osobnym procesie, ocenę i diagnostykę. Przechodzi 20 z 20 testów, a dwa kolejne przebiegi dają identyczne pliki. Model to regresja logistyczna L2 z C = 10 i progiem 0,329, wybrany na train. Zapisany JSON zawiera reguły przetwarzania, słownik (2632 słowa), IDF, wagi i próg.

Przewidywanie z tego pliku na kopii validation bez kolumny `label` daje te same oceny co przebieg uczący. Sprawdziłem to też poza projektem, w Pythonie 3.13: 60 z 60 ocen identycznych. Sumy SHA-256 modelu zapisane przed otwarciem validation zgadzają się z sumami po ocenie.

| | TP / FP / FN / TN | fałszywe alarmy | czułość | precyzja |
|---|---|---|---|---|
| CV na train (180) | 58 / 3 / 2 / 117 | 3/120 | 58/60 | 58/61 |
| validation (60), model | 19 / 0 / 1 / 40 | 0/40 | 19/20 | 19/19 |
| validation, „zawsze ham” | 0 / 0 / 20 / 40 | 0/40 | 0/20 | – |
| validation, reguła `<MONEY>` + `remove` | 5 / 1 / 15 / 39 | 1/40 | 5/20 | 5/6 |
| validation, Naiwny Bayes (informacyjnie) | 19 / 0 / 1 / 40 | 0/40 | 19/20 | 19/19 |

Macierz pomyłek dla validation przeliczyłem niezależnie z pliku przewidywań i etykiet. Jedyny błąd to francuskojęzyczny newsletter (`017f0817d4c92951`) z oceną 0,217. Większość jego słów nie występuje w słowniku z train, a kwota „0.34 Euros” nie daje tokenu `<MONEY>`. Najwyżej oceniony ham ma 0,200, więc margines jest bardzo mały. Progi 0,25, 0,40 i 0,50 dają ten sam wynik.

**Najważniejsze odstępstwa od planu:**
- TF-IDF, regresję (metoda Newtona zamiast L-BFGS, który nie zbiegł) i Naiwnego Bayesa napisaliśmy sami w bibliotece standardowej.
- Maskowanie rozszerzyliśmy na bramce o nagłówki kanałów RSS (`URL:` 33 ham / 0 spam, `Date:` 35 ham / 0 spam), bo były najsilniejszymi wagami ham.
- Poprawiliśmy wzorce banera Yahoo i stopek Mailmana.
- Znaczniki list usuwamy w całym tekście, także w cytowanych tematach.
- Słowo `remove` w adresach (`remove.php`) znika w tokenie `<URL>`. W tokenach zostaje 17 z 24 spamów, które je zawierają.

**Co pokazała diagnostyka:**
- Po usunięciu nagłówków RSS Naiwny Bayes spadł w CV z 60/60 do 57/60 przy 3 fałszywych alarmach, a regresja została przy 58/60. Jego wcześniejsza przewaga opierała się więc na skrócie źródła.
- Porównanie z maskowaniem i bez daje ten sam wynik (validation 19/20, 0/40). Nie dowodzi to, że maskowanie jest zbędne: validation pochodzi z tych samych archiwów, więc skrót działa w niej tak samo jak w train.
- Dopisanie maskowanej stopki ILUG do 20 spamów zmieniło ocenę w 0 z 20 przypadków. Dopisanie linii „On …, David wrote:” z cytatem nie zmieniło żadnej decyzji, ale obniżyło ocenę o medianę 0,058 (maksymalnie 0,18). Model nadal reaguje na styl odpowiedzi z list.
- Wspólne wątki: 7 wiadomości (1 spam, 6 ham), dokładność 7/7, pozostałe 52/53. Nie widać zawyżenia, ale 7 wiadomości to za mało, żeby to wykluczyć.
- Na ocenach z CV udział spamu rośnie razem z oceną: od 1/106 poniżej 0,2 do 36/36 powyżej 0,8. Próg 0,329 leży w przedziale 0,2–0,4, w którym spam to 3/15.

**Ograniczenia i czego nie sprawdzono:**
- Validation ma 60 wiadomości, więc 1 spam to 5 punktów procentowych czułości, a 1 ham to 2,5 punktu odsetka fałszywych alarmów. Wynik 19/20 i 0/40 nie odróżnia regresji od Naiwnego Bayesa.
- Dane to poczta z 2002 roku, tylko easy_ham, w proporcji 2:1. Precyzja przy realnym, mniejszym udziale spamu będzie niższa.
- W wagach ham zostały słowa związane z archiwami list (`wrote`, `cvs`, `rpm`, `exmh`, `linux`, `david`, `spam`) oraz `sep` z linii przypisania cytatu (12 ham / 0 spam). Tych linii świadomie nie maskowaliśmy.
- Nie sprawdzono: poczty w innych językach, wiadomości HTML i wieloczęściowych, trudniejszego ham (hard_ham), pochodzenia tokenów `ne` i `su` w wagach spamu ani wariantu z maskowaniem linii cytatu i rozszerzonym `<MONEY>` (np. „Euros”). Takie zmiany trafiłyby do następnej wersji i wymagałyby nowych danych do oceny, bo validation została już użyta.
