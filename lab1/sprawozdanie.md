Repozytorium: https://github.com/krzzbi/miob_lab (folder labu: [lab1](https://github.com/krzzbi/miob_lab/tree/main/lab1))

Antigravity · Gemini 4 Argon · Medium

# Laboratorium 1 — pierwsze uruchomienie projektu z agentem

## 1. Jaki wynik zlecono, jakie zmiany były dozwolone i o jakie informacje do odbioru poproszono agenta?

Przed uruchomieniem czegokolwiek ustaliłem z agentem zlecenie:

- **Wynik:** raport dla `data/exercise.csv` zapisany w `results/exercise-summary.json` (folder `lab1/` pełni rolę `project/` z treści zadania).
- **Granice:** w etapie 1 bez zmian kodu (`describe_data.py`), danych (`data/exercise.csv`) i testów (`tests/`). W etapie 2 dozwolona zmiana jednej etykiety i odświeżenie raportu — zlecana osobno.
- **Odbiór:** agent ma wskazać miejsce raportu, odczytać jego treść, pokazać użyty plik wejściowy, dokładne wykonane polecenie i wynik procesu (kod wyjścia), podać wynik testu i zgłosić niewykonane kroki.

Agent potwierdził zakres i zaproponował dodatkowo sprawdzenie katalogu roboczego oraz wersji Pythona przed uruchomieniem. Przed zleceniem wykonania przyjąłem ten zakres razem z propozycją agenta, bo przed uruchomieniem chciałem wiedzieć, w jakim katalogu i na jakim interpreterze wykonują się polecenia (na moim komputerze czy w innym środowisku aplikacji). Dodatkowo poprosiłem o sprawdzenie, czy raport istnieje przed uruchomieniem, żeby odróżnić nowy wynik od ewentualnej wcześniejszej próby.

Wcześniej, na podstawie README, agent wyjaśnił działanie `describe_data.py` (czyta CSV, zlicza wiadomości i etykiety `ham`/`spam`, zapisuje raport JSON; nie trenuje modelu) i wskazał instrukcję uruchomienia (README, sekcja *Environment and first run*, pkt 3). W sesji widać odczyt `README.md` i `data/exercise.csv`. Na przykładzie wiadomości `exercise-001` (etykieta `ham`) ustaliłem znaczenie pól: `message_id` — identyfikator łączący rekord z oryginałem `data/raw/<message_id>.eml`; `text` — temat i pełna treść, może zajmować wiele linii pliku (dlatego liczba linii ≠ liczba wiadomości); `label` — etykieta ze zbioru (`ham` lub `spam`).

## 2. Jak powstał raport?

- Wejście: `data/exercise.csv`
- Polecenie: `/Users/jasminetea/.pyenv/shims/python3 describe_data.py --data data/exercise.csv --out results/exercise-summary.json`
- Narzędzie aplikacji: `run_command` (polecenia w terminalu), `read_file` (odczyt raportu)
- Środowisko wykonania: mój komputer (macOS, lokalna powłoka `zsh` uruchamiana przez Antigravity), katalog roboczy `/Users/jasminetea/Coding/WATs3/miob_lab/lab1`, Python 3.12.1 (`/Users/jasminetea/.pyenv/versions/3.12.1/bin/python3`)
- Wynik procesu: kod wyjścia 0, stdout: `Report saved: …/results/exercise-summary.json` i `Messages: 8; labels: {'ham': 5, 'spam': 3}`
- Raport z etapu 1: `total_messages` = 8, `ham` = 5, `spam` = 3 — w pliku jest 8 rekordów (wiadomości, nie linii), z czego 5 ma etykietę `ham`, a 3 etykietę `spam`. `input_file` wskazuje na `data/exercise.csv`, czyli plik podany w `--data`.
- **Końcowy raport (po etapie 2): [results/exercise-summary.json](results/exercise-summary.json)** — `total_messages` = 8, `ham` = 4, `spam` = 4. Nadal 8 wiadomości, ale po zmianie etykiety `exercise-001` z `ham` na `spam` jest 4 `ham` i 4 `spam` (szczegóły w punkcie 5).

Skąd wiem, że raport pochodzi z tego uruchomienia: przed pierwszym uruchomieniem plik nie istniał (`ls` zwrócił kod 1). Pierwsze uruchomienie agent wykonał systemowym `/usr/bin/python3` w wersji 3.9.6, bo jego powłoka miała inny `PATH` niż mój terminal, a README wymaga Pythona 3.10+. Dlatego powtórzyłem uruchomienie, podając pełną ścieżkę do Pythona 3.12.1 z pyenv (pierwsza próba zakończyła się kodem 127, bo dostęp do `~/.pyenv` wymagał mojej zgody w Antigravity; po zgodzie polecenie się wykonało). Nowe uruchomienie nadpisało raport — sekcja `runtime` pokazuje teraz Pythona 3.12.1, a liczebności i `input_sha256` są takie same, bo dane się nie zmieniły.

Podział ról: model wygenerował wywołania narzędzi (`run_command` z poleceniem, `read_file` ze ścieżką raportu); Antigravity obsłużyła narzędzie, poprosiła o zgodę na dostęp poza folder projektu i przekazała polecenie do lokalnej powłoki; mój komputer uruchomił interpreter, który przeczytał CSV, policzył etykiety i zapisał JSON; wynik procesu (stdout, kod wyjścia) i treść pliku wróciły do modelu jako odpowiedzi narzędzi. Sam tekst komendy w odpowiedzi agenta nie potwierdza wykonania — potwierdzają je wywołania narzędzi i ich wyniki widoczne w sesji.

## 3. Który fragment dokumentacji lub kodu wyjaśniłeś?

- Plik i miejsce: `README.md` — „another successful run with the same `--out` replaces the report” oraz informacja, że raport nie aktualizuje się sam po zmianie danych; odpowiadające miejsce w `describe_data.py`: blok `try … except` w funkcji `main` (linie 55–62) — `describe(args.data)` jednorazowo czyta i waliduje CSV, a dopiero po jego powodzeniu `args.out.write_text(...)` zapisuje raport, nadpisując poprzednią zawartość pliku; błąd danych kończy program (`parser.exit(1, …)`) przed zapisem
- Moje wyjaśnienie: raport to plik zapisany w chwili uruchomienia skryptu — jeśli potem zmienię CSV, raport dalej pokazuje stare liczby, dopóki nie uruchomię skryptu ponownie. Kolejne udane uruchomienie z tym samym `--out` nadpisuje plik w całości, a jeśli uruchomienie się nie powiedzie, zapis nie następuje i na dysku zostaje stary raport.
- Zmieniony przypadek (zaproponowany przez agenta, rozważany tylko na papierze — bez zmian w plikach): (1) udane uruchomienie z `--out results/raport.json` na danych 7× `ham`, 3× `spam`; (2) zmiana jednej etykiety z `ham` na literówkę `spamm`; (3) ponowne uruchomienie tej samej komendy. Pytanie: jaki kod wyjścia i co będzie w `results/raport.json`?
- Przewidywanie: etykieta `spamm` nie przejdzie walidacji, program zakończy się kodem wyjścia 1 z komunikatem `Error: …`, zapis nie nastąpi, więc `results/raport.json` nadal będzie istniał z danymi z kroku 1 (`total_messages` = 10, `ham` = 7, `spam` = 3) — czyli nie będzie odpowiadał aktualnym danym.
- Sposób sprawdzenia: agent zweryfikował przewidywanie w kodzie `describe_data.py`, bez uruchamiania: linia 19 — słownik `counts` ma tylko klucze `ham` i `spam`; linie 26–27 — nieznana etykieta powoduje `ValueError("Record …: label must be ham or spam.")`; linia 56 — `describe()` wykonuje się przed zapisem `write_text` (linie 58–60), więc wyjątek przerywa blok `try` przed zapisem; linie 61–62 — wyjątek jest łapany, a `parser.exit(1, …)` kończy proces z kodem 1.
- Wniosek: przewidywanie było w pełni zgodne z kodem. Sama obecność pliku raportu nie dowodzi, że ostatnie uruchomienie się powiodło ani że raport opisuje aktualne dane (README: *„An older report may still exist, so its presence alone does not prove that the latest run succeeded.”*). Dlatego trzeba sprawdzać kod wyjścia i wynik konkretnego uruchomienia w sesji, a po zmianie danych uruchomić skrypt ponownie.

## 4. Jak uruchomiono gotowy test, jaki był wynik i czego dotyczyło sprawdzenie?

- Polecenie: `/Users/jasminetea/.pyenv/shims/python3 -m unittest discover -s tests -v` (zgodnie z README), uruchomione przez agenta narzędziem `run_command`.
- Wynik: `test_cli_writes_expected_counts … ok`, `Ran 1 test`, `OK`, kod wyjścia 0 (taki sam wynik dał wcześniej Python 3.9.6).
- Czego dotyczyło sprawdzenie: test uruchamia `describe_data.py` na trzech syntetycznych wiadomościach z `tests/fixtures/messages.csv` (2× `ham`, 1× `spam`, treści wieloliniowe) i w katalogu tymczasowym sprawdza, że program kończy się sukcesem, zapisuje czytelny raport JSON, wskazuje właściwy plik wejściowy i podaje dokładnie 3 / 2 / 1. Oczekiwane liczby są wpisane w test na stałe (`tests/test_report.py`, linie 30–32), a nie liczone przez testowany program, więc test porównuje wynik skryptu z niezależnym wzorcem. Plik fixture ma 10 fizycznych linii, ale tylko 3 wiadomości — test sprawdza więc także, że skrypt liczy rekordy CSV, a nie linie.
- Czego test nie potwierdza: nie czyta mojego `results/exercise-summary.json`, więc nie potwierdza liczebności dla `data/exercise.csv` — to potwierdza odczyt raportu z mojego uruchomienia. Nie dowodzi też poprawności dla każdego zbioru danych ani jakości przyszłego modelu ML.

## 5. Jaką etykietę zmieniono w etapie 2?

- Zmieniona etykieta: wiadomość `exercise-001` w `data/exercise.csv`, `ham` → `spam` (celowa zmiana na potrzeby ćwiczenia). Najpierw zleciłem wyłącznie tę zmianę — bez uruchamiania programu i testu, bez zmian w raporcie i innych plikach. Agent pokazał zmieniony wiersz (`",ham` → `",spam`); `git diff` nie zadziałał (kod 128), bo folder nie był rozpoznany jako repozytorium.
- Oczekiwana zmiana (zapisana w sesji przed wykonaniem): dotychczasowy raport nie opisuje aktualnych danych, bo powstał przed zmianą i nie aktualizuje się sam. Oczekiwałem `ham` 5 → 4, `spam` 3 → 4, `total_messages` bez zmian (8) oraz nowego `input_sha256`, bo zmieniły się bajty pliku; `input_file` i `runtime` bez zmian.
- Wybrane operacje i uzasadnienie: (1) ponowne uruchomienie `describe_data.py` z tym samym `--out` i sprawdzenie kodu wyjścia — tylko nowe, udane uruchomienie tworzy raport z aktualnych danych; (2) odczyt nowego raportu i porównanie z przewidywaniem oraz z raportem sprzed zmiany. Nie powtarzałem testu: test nie czyta `exercise-summary.json`, a kod się nie zmienił, więc wcześniejszy wynik `OK` nie potwierdza aktualnych liczebności i ponowny test też by ich nie potwierdził. Agent ocenił plan jako kompletny.
- Wykonanie: `/Users/jasminetea/.pyenv/shims/python3 describe_data.py --data data/exercise.csv --out results/exercise-summary.json` — kod wyjścia 0, stdout: `Messages: 8; labels: {'ham': 4, 'spam': 4}` (pierwsza próba zakończyła się kodem 127 do czasu mojej zgody w Antigravity na dostęp do `~/.pyenv`).

| Pole | Przed zmianą (etap 1) | Po nowym uruchomieniu (etap 2) | Zgodne z przewidywaniem |
|---|---|---|---|
| `input_file` | `…/lab1/data/exercise.csv` | `…/lab1/data/exercise.csv` | tak (bez zmian) |
| `input_sha256` | `95e9049f…ed6ed07` | `7dc8080d…18b8152` | tak (zmieniony) |
| `total_messages` | 8 | 8 | tak |
| `ham` | 5 | 4 | tak |
| `spam` | 3 | 4 | tak |
| `runtime` | Python 3.12.1, `…/lab1` | Python 3.12.1, `…/lab1` | tak (bez zmian) |

- Z którego uruchomienia pochodzi wynik: z uruchomienia w etapie 2 zakończonego kodem 0 — zgadzają się ze sobą stdout tego procesu (`ham: 4, spam: 4`), nowy `input_sha256` i treść odczytanego pliku. Liczebności w raporcie nie były poprawiane ręcznie. Treść raportu sprzed zmiany i po niej jest widoczna w sesji.
