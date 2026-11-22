# Laboratorium 2 — Od problemu do planu rozwiązania

**Repozytorium:** https://github.com/krzzbi/miob_lab

**Agent:** Antigravity · Gemini 4 Argon · poziom rozumowania: Medium

---

## 1. Cel i zakres

Celem jest plan lokalnego prototypu, który na podstawie tekstu wiadomości e-mail (temat i treść) ocenia, czy jest ona spamem.

- **Wejście:** tekst wiadomości, opcjonalnie z `message_id`, gdy wiadomość pochodzi z pliku CSV.
- **Wyjście:** dla każdej wiadomości jej `message_id`, przewidywana etykieta (`spam` albo `ham`) i ocena liczbowa od 0 do 1, z której wynika etykieta po porównaniu z progiem.
- **Zastosowanie:** sprawdzenie, czy podejście oparte na treści wiadomości odróżnia spam od prawidłowej poczty na dostarczonych danych z Apache SpamAssassin. Prototyp nie łączy się ze skrzynką, nie przenosi, nie usuwa i nie wysyła wiadomości, nie otwiera zawartych w nich linków i nie zmienia dostarczonych plików.

Plan kończy się na etapie koncepcji. Uczenie modelu i pomiar jakości są opisane jako przyszłe kroki. Wynik na tym zbiorze dotyczy historycznej poczty z 2002 roku i proporcji klas ustalonej na potrzeby zajęć (2:1), więc nie potwierdzi skuteczności na współczesnej poczcie.

## 2. Wybrane podejście i uzasadnienie

### 2.1. Dane i ich przygotowanie

Pojedynczy przykład to jeden wiersz CSV, czyli jedna wiadomość e-mail: `message_id` (identyfikator do śledzenia wyników, nie jest wejściem modelu), `text` (temat w pierwszej linii `Subject: ...`, pusta linia, treść) oraz `label` (`spam`/`ham`, etykieta z korpusu). Etykieta jest potrzebna przy uczeniu (model dopasowuje się do znanych odpowiedzi) i przy ocenie (porównanie przewidywań z prawdą), ale nie przy klasyfikowaniu nowej wiadomości.

Obserwacje z danych (sprawdzone niezależnie od agenta):
- `train.csv`: 180 wiadomości (120 ham, 60 spam), `validation.csv`: 60 (40 ham, 20 spam), czyli w obu zbiorach proporcja 2:1. Liczności i sumy SHA-256 plików zgadzają się z `manifest.json`.
- Brak pustych tekstów i etykiet, etykiety tylko `spam`/`ham`. Brak powtórzeń `message_id` i tekstów (także po zamianie na małe litery i ujednoliceniu białych znaków), zarówno w obrębie plików, jak i między train a validation. Zgadza się to z dokumentacją, która opisuje deduplikację przed podziałem.
- `text` nie zawiera surowych nagłówków transportowych (`Received:`, `Content-Type:`). HTML występuje tylko sporadycznie, bo wybrano wiadomości o pojedynczej części tekstowej. Adresy URL i e-mail są częste w obu klasach, a adresy są zanonimizowane (np. `zzzz@spamassassin.taint.org`).
- Różnice między klasami: spam ma większą medianę długości (ok. 1430 vs 755 znaków w train), więcej wielkich liter i kwot pieniężnych. Ham znacznie częściej zawiera cytowania `>` (ok. 47% ham vs ok. 4% spam).
- Znaczniki list dyskusyjnych w temacie występują w podobnym odsetku obu klas (ok. 30%), ale prawie każda konkretna nazwa listy należy tylko do jednej klasy: `[Spambayes]` 0 spam / 11 ham, `[SAtalk]` 0/10, `[ILUG-Social]` 8/0. Mieszane jest tylko `[ILUG]` (15/7). Podobnie działają stopki list i domeny (`newsisfree.com` 0/22, `freshrpms.net` 0/14) oraz ślady anonimizacji korpusu (`zzzz@spamassassin.taint.org` 7/0). Te elementy wskazują, z jakiej listy lub archiwum pochodzi wiadomość, a nie czy jest spamem. Model mógłby się ich nauczyć jako „skrótu”. Validation pochodzi z tych samych archiwów, więc taki skrót dawałby tam dobry wynik, ale zawiódłby na innej poczcie.

Decyzje dotyczące przygotowania:
- **Nie usuwamy wierszy i nie poprawiamy etykiet.** Nie ma braków ani powtórzeń, a podział 180/60 ma zostać zachowany. Oryginalne pliki pozostają bez zmian, a wszystkie przekształcenia są krokiem przetwarzania wykonywanym na kopii w pamięci.
- **Rozdzielamy kolumny według roli.** Wejściem modelu jest tylko `text`. `label` służy do uczenia (train) i do oceny (validation). `message_id` zachowujemy, żeby powiązać przewidywania z wiadomościami przy analizie błędów.
- **Maskujemy artefakty źródła** z ustalonej z góry listy wzorców: znaczniki nazw list w temacie, stopki doklejane przez oprogramowanie list oraz ślady anonimizacji korpusu (`zzzz`, `spamassassin.taint.org`). Nie usuwamy natomiast ogólnych zwrotów typu „unsubscribe” czy „remove me” z treści spamu, bo są częścią samego spamu, a nie śladem źródła: słowo `remove` (i jego odmiany) występuje w 43 z 80 wiadomości spam i tylko w 5 ze 160 ham. W ham `unsubscribe` pojawia się głównie w automatycznej stopce jednej listy (9 z 12 przypadków w train), więc po zamaskowaniu stopek ten sygnał jest jeszcze czystszy. Wzorce ustalamy na podstawie train i dokumentacji i nie dostrajamy ich pod wynik na validation.
- Stały prefiks `Subject:` występuje w każdej wiadomości, więc nie odróżnia klas i można go pominąć. Temat i treść przekazujemy dalej razem.

Uwaga: w rzeczywistym filtrze osobistym przynależność do listy mogłaby być uczciwą wskazówką (moje listy to zwykle ham). Tutaj maskujemy ją, bo celem jest rozpoznawanie spamu po treści, a ocena na validation z tych samych archiwów nie wykryłaby, że model opiera się na nazwie listy.

### 2.2. Reprezentacja wiadomości

Model potrzebuje liczb, więc każdą wiadomość zamieniamy na wektor **TF-IDF pojedynczych słów**. Każda pozycja wektora odpowiada jednemu słowu ze słownika. Jej wartość jest tym większa, im częściej słowo występuje w tej wiadomości i im rzadziej w całym zbiorze uczącym. Wektor jest normalizowany, więc długie wiadomości nie dominują samą objętością.

Przetwarzanie przed zbudowaniem wektora (po przygotowaniu z 2.1):
1. Każde słowo pisane w całości wielkimi literami (co najmniej 3 litery) oznaczamy dodatkowym tokenem `<CAPS>`, a każdy wykrzyknik zamieniamy na token `<EXCLAM>`.
2. Zamieniamy tekst na małe litery, a adresy URL, adresy e-mail, kwoty pieniężne i pozostałe liczby na tokeny `<URL>`, `<EMAIL>`, `<MONEY>`, `<NUM>`.
3. Dzielimy tekst na słowa. Do słownika trafiają tylko słowa występujące w co najmniej 2 wiadomościach z train.

Uzasadnienie z danych (mała próba, tylko odczyt, bez zapisu plików):
- Słownik z train ma ok. 7700 słów, ale 62% z nich występuje tylko w jednej wiadomości. Po odcięciu zostaje ok. 2650–2900 słów, a w validation wciąż ok. 80% wystąpień słów jest znanych. Reprezentacja słownikowa jest więc wykonalna przy 180 przykładach. Gdyby nieznanych słów w validation było większość, rozważylibyśmy n-gramy znakowe.
- Tokeny zastępcze: kwoty występują w 36 z 60 wiadomości spam i w 6 ze 120 ham w train. Surowe liczby rozproszyłyby ten sygnał na dziesiątki różnych wartości. Konkretne adresy tworzyłyby jednorazowe słowa albo ślady źródła (`newsisfree.com`).
- `<CAPS>` i `<EXCLAM>`: mediana słów pisanych wielkimi literami to 9,5 w spamie i 1 w ham, średnia liczba wykrzykników to 6,8 i 0,6. Zamiana na małe litery bez tych tokenów usunęłaby ten sygnał.

Rozważane alternatywy:
- Doklejenie tych samych sygnałów jako osobnych liczb (długość, udział wielkich liter itd.) wymagałoby osobnego skalowania dopasowanego na train. Utrudniłoby też uczciwe porównanie modeli, bo Naiwny Bayes słabo pasuje do takich cech. Tokeny w tym samym worku słów dają obu modelom identyczne wejście.
- Gotowe osadzenia z modelu językowego wymagają pobrania dużego modelu, a ich decyzje trudniej przejrzeć pod kątem skrótów.
- Same n-gramy znakowe dają bardzo dużo cech przy 180 przykładach.

Co reprezentacja zachowuje: słownictwo powtarzające się w danych uczących, to, jak typowe lub rzadkie jest słowo, zagęszczenie linków, adresów, kwot i liczb, „krzykliwość” wiadomości.
Co pomija: kolejność słów i kontekst, słowa jednorazowe z train i słowa nieznane z nowych wiadomości (są ignorowane), konkretne adresy i wartości liczbowe, bezwzględną długość wiadomości. Tę ostatnią tracimy świadomie: w danych spam jest przeciętnie dłuższy, ale to słaba i niestabilna wskazówka.

Cytowania `>` (47% ham, ok. 4% spam) nie dostają osobnego tokenu. W tym zbiorze w dużej mierze odzwierciedlają źródło (dyskusje na listach w easy_ham), a reguła „cytat = ham” byłaby łatwa do obejścia.

Dopasowywane tylko na train i zachowywane do klasyfikowania nowych wiadomości: **słownik** (słowo → pozycja w wektorze) i **wagi IDF**. Reguły z kroków 1–2 są stałe, ale muszą być stosowane identycznie do train, validation i nowych wiadomości.

### 2.3. Model i porównanie z alternatywą

Model to reguła z parametrami dopasowanymi do przykładów. W naszym przypadku są to wagi przypisane każdemu słowu ze słownika i wyraz wolny. Podczas uczenia wagi słów częstych w wiadomościach z etykietą spam (np. `remove`, `<MONEY>`) rosną, a słów typowych dla ham (np. `cvs`, `perl`) maleją, tak żeby odpowiedzi modelu na train jak najlepiej zgadzały się z etykietami.

**Model główny: regresja logistyczna z regularyzacją L2.** Sumuje wagi słów obecnych w wiadomości i zamienia wynik na liczbę od 0 do 1 (ocenę, że wiadomość to spam). Wiadomość jest uznawana za spam, gdy ocena przekracza próg.
- Pasuje do reprezentacji TF-IDF i radzi sobie ze skorelowanymi tokenami: w spamie `<CAPS>`, `<EXCLAM>` i `<MONEY>` występują razem, a model dzieli między nie wagę zamiast liczyć ten sam dowód kilka razy.
- Słów jest więcej (ok. 2650) niż przykładów (180), więc bez regularyzacji model łatwo dopasowałby się do pojedynczych wiadomości. Regularyzacja L2 karze duże wagi i wygładza model. Jej siłę trzeba wybrać.
- Zwraca ocenę, przy której można przesuwać próg. To ważne, bo prawidłowa wiadomość uznana za spam jest zwykle bardziej kosztowna niż przeoczony spam.
- Wagi można odczytać, co jest potrzebne do sprawdzenia, czy model nie korzysta ze skrótów źródła (sekcja 5).

**Alternatywa: wielomianowy Naiwny Bayes.** Dla każdej klasy szacuje, jak często pojawiają się w niej poszczególne słowa, i wybiera klasę, w której słowa wiadomości są bardziej prawdopodobne. Uczy się przez proste zliczanie, jest bardzo szybki i ma tylko jedno ustawienie (wygładzanie). Przy małej próbie bywa stabilniejszy i jest klasycznym punktem odniesienia dla filtrów spamu. Słabsze strony: zakłada niezależność słów, przez co powiązane sygnały spamu liczy wielokrotnie, a jego oceny są zwykle skrajne (bliskie 0 albo 1), co utrudnia sensowne przesuwanie progu. Oba modele dostają identyczne wejście, więc porównanie dotyczy tylko samego modelu.

**Wybór ustawień bez dotykania validation.** Siłę regularyzacji (dla Naiwnego Bayesa: wygładzanie) i próg decyzji wybieramy 5-krotną walidacją krzyżową na train z zachowaniem proporcji klas. Każda część ma 36 wiadomości (24 ham, 12 spam). Słownik i wagi IDF są budowane od nowa w każdym przebiegu tylko na części uczącej. Ograniczenie: przy 12 spamach w części jedna pomyłka zmienia wynik o ok. 8 punktów procentowych. Dlatego sprawdzamy tylko kilka zgrubnych wartości ustawień. Po wyborze model uczony jest raz na całym train.

Naiwnego Bayesa wybralibyśmy zamiast regresji, gdyby w walidacji krzyżowej na train wykrywał wyraźnie więcej spamu przy tej samej liczbie fałszywych alarmów albo dawał wyraźnie stabilniejsze wyniki między częściami. Inny powód: gdyby regresja opierała się na pojedynczych rzadkich słowach specyficznych dla źródła. Decyzję podejmujemy na train. Validation służy tylko do końcowej oceny wybranego wariantu.

## 3. Kroki planu i ich zależności

| Krok | Dane wejściowe | Co powstaje | Do czego jest potrzebne dalej |
|---|---|---|---|
| 1. Wczytanie danych | `train.csv`, `validation.csv` (bez zmian) | kopie w pamięci z rozdzielonymi kolumnami `text`, `label`, `message_id` | `text` do przetwarzania, `label` do uczenia i oceny, `message_id` do powiązania wyników z wiadomościami |
| 2. Przygotowanie tekstu | `text` | tekst bez artefaktów źródła, z tokenami `<CAPS>`, `<EXCLAM>`, `<URL>`, `<EMAIL>`, `<MONEY>`, `<NUM>`, podzielony na słowa | budowa słownika i wektorów. Reguły są stałe i muszą być identyczne dla train, validation i nowych wiadomości |
| 3. Wybór ustawień i modelu | przygotowane teksty i etykiety z train | dla każdego kandydata (regresja logistyczna z kilkoma wartościami siły regularyzacji, Naiwny Bayes z kilkoma wartościami wygładzania): wyniki 5-krotnej walidacji krzyżowej, w której słownik, IDF i model powstają od nowa na części uczącej, oraz oceny dla wiadomości pominiętych w danym przebiegu | wybór modelu, jego ustawień i progu decyzji. Wybór zapada tylko na train |
| 4. Uczenie końcowe | wszystkie 180 wiadomości z train z etykietami, wybrane ustawienia | słownik (słowa występujące w ≥ 2 wiadomościach), wagi IDF, wagi modelu i wyraz wolny | klasyfikacja validation i nowych wiadomości |
| 5. Klasyfikacja validation | teksty z validation **bez etykiet**, zachowane reguły, słownik, IDF, wagi, próg | dla każdej wiadomości `message_id`, ocena i przewidywana etykieta | ocena (krok 6). Słownika ani modelu nie dopasowuje się tu od nowa |
| 6. Ocena | przewidywania z kroku 5 i etykiety z validation, połączone po `message_id` | macierz pomyłek, miary, lista błędnie zaklasyfikowanych wiadomości | wnioski i sprawdzenia z sekcji 5 |

**Co zachować:**
- *Do klasyfikowania nowych wiadomości bez ponownego uczenia:* reguły przetwarzania w ustalonej kolejności, słownik, wagi IDF i sposób normalizacji, wagi modelu z wyrazem wolnym, próg decyzji.
- *Do oceny i odtworzenia wyników:* oryginalne pliki danych, wybrane ustawienia i wyniki walidacji krzyżowej oraz przewidywania na validation razem z `message_id` i ocenami.

Do samej klasyfikacji nie są potrzebne dane uczące ani macierz TF-IDF zbioru uczącego.

**Uczenie a klasyfikacja nowej wiadomości.** Uczenie wymaga etykiet i zmienia parametry: słownik, IDF i wagi. Klasyfikacja nowej wiadomości nie potrzebuje etykiety i niczego nie zmienia. Wiadomość przechodzi przez te same reguły przetwarzania, jest zamieniana na wektor z użyciem zachowanego słownika i IDF (nieznane słowa są pomijane), model liczy ocenę, a próg wyznacza etykietę. Gdyby słownik zbudowano od nowa albo przetwarzanie różniło się od tego z uczenia, pozycje wektora oznaczałyby inne słowa niż te, dla których wyuczono wagi, albo cechy takie jak `<MONEY>` nigdy by się nie pojawiły. Odpowiedzi modelu byłyby wtedy bezwartościowe.

**Etykiety są potrzebne** przy uczeniu i wyborze ustawień (train) oraz przy ocenie (validation, dopiero po wygenerowaniu przewidywań). Nie są potrzebne przy klasyfikacji nowej wiadomości.

## 4. Planowana ocena

**Dane.** Ocena odbywa się raz, na końcu, na `validation.csv` (60 wiadomości: 40 ham, 20 spam). Te wiadomości nie biorą udziału w budowie słownika, uczeniu, wyborze ustawień, wyborze progu ani wyborze między modelami. Gdyby brały, wynik mierzyłby dopasowanie do tych konkretnych 60 wiadomości i byłby zawyżony względem nowych wiadomości. Wszystkie decyzje zapadają wcześniej, na train (sekcja 2.3).

**Porównanie przewidywań z etykietami.** Model klasyfikuje teksty z validation bez dostępu do etykiet. Następnie przewidywania łączymy z etykietami po `message_id` i każdą wiadomość przypisujemy do jednej z czterech grup (spam jest klasą „pozytywną”):

| | przewidziano spam | przewidziano ham |
|---|---|---|
| **naprawdę spam (20)** | TP: wykryty spam | **FN: przeoczony spam** |
| **naprawdę ham (40)** | **FP: prawidłowa wiadomość uznana za spam** | TN: poprawnie przepuszczona wiadomość |

**Miary** (zawsze z liczbami bezwzględnymi obok procentów, np. „17/20”):
- **Odsetek fałszywych alarmów** = FP / liczba ham. Pokazuje, jaka część prawidłowej poczty trafiłaby do spamu. To najważniejsza miara, bo ten błąd jest najbardziej kosztowny.
- **Czułość dla spamu** = TP / liczba spam. Pokazuje, jaka część spamu została wykryta. Jej dopełnienie to odsetek przeoczonego spamu (FN).
- **Precyzja dla spamu** = TP / wszystkie wiadomości oznaczone jako spam. Pokazuje, ile wśród oznaczonych jest fałszywych alarmów. Zależy od udziału spamu w danych (patrz sekcja 5).
- **Pole pod krzywą precyzja–czułość** (pomocniczo). Ocenia, czy model nadaje spamowi wyższe oceny niż ham niezależnie od wybranego progu.
- **Dokładność** tylko pomocniczo. Model „zawsze ham” ma na validation dokładność 40/60 ≈ 67% przy zerowym wykryciu spamu, a dokładność nie odróżnia fałszywego alarmu od przeoczonego spamu.

**Mała liczebność.** Jedna przeoczona wiadomość spam zmienia czułość o 5 punktów procentowych (1/20), a jeden fałszywy alarm zmienia jego odsetek o 2,5 punktu (1/40). Różnic rzędu 1–2 wiadomości (np. między modelami) nie traktujemy jako dowodu przewagi.

**Wybór progu (na train).** Na ocenach z walidacji krzyżowej (wszystkie 180 wiadomości, każda oceniona przez model, który jej nie widział) wybieramy próg dający największą czułość dla spamu przy co najwyżej 3 fałszywych alarmach na 120 wiadomości ham (≤ 2,5%). Wymóg zera fałszywych alarmów byłby zbyt wrażliwy na pojedynczą nietypową wiadomość i wypchnąłby próg bardzo wysoko.

**Punkty odniesienia** (tak samo oceniane na validation):
- model „zawsze ham” (zero fałszywych alarmów, zero wykrytego spamu);
- prosta reguła ustalona na train (np. wiadomość z `<MONEY>` i słowem `remove` to spam), żeby sprawdzić, czy model wnosi coś ponad oczywistą heurystykę;
- alternatywa: Naiwny Bayes. Wynik pokazujemy informacyjnie, bez zmiany wcześniej podjętej decyzji.

**Analiza poza liczbami.** Przeglądamy po `message_id` każdy fałszywy alarm i każdy przeoczony spam: treść wiadomości i słowa, które najbardziej wpłynęły na ocenę. Szukamy przyczyny, np. wiadomość ham o pieniądzach, krótki spam bez znanych słów albo ślad źródła.

## 5. Niewiadome i możliwe korekty

**Czy model mimo maskowania korzysta ze skrótów źródła?** Validation pochodzi z tych samych archiwów co train, więc sam wynik na niej tego nie pokaże. Sprawdzenie po uczeniu:
- przejrzeć 20–30 słów najmocniej wpływających na decyzję „spam” i „ham” i zobaczyć, czy są wśród nich nazwy list, domen lub artefaktów korpusu (np. `newsisfree`, `freshrpms`, `sourceforge`);
- na kopiach kilku wiadomości z validation usunąć lub dopisać ślad źródła (np. stopkę listy) i sprawdzić, czy zmienia to decyzję modelu;
- porównać model uczony z maskowaniem i bez niego. Wynik tego porównania służy do diagnozy, a nie do wyboru wariantu z wyższym wynikiem.

Jeśli ślady źródła nadal dominują, trzeba rozszerzyć listę maskowanych wzorców (na podstawie train) i powtórzyć uczenie.

**Czy między train a validation są wspólne wątki?** Dokumentacja zaznacza, że tego nie sprawdzono w pełni. Wstępne sprawdzenie tematów (po usunięciu `Re:`/`Fwd:` i znaczników list) pokazuje, że 7 z 60 wiadomości validation (6 ham, 1 spam) ma ten sam temat co wiadomość z train. Te wiadomości mogą dzielić cytowany tekst, więc wynik na validation może być zawyżony. Sprawdzenie: policzyć wyniki osobno dla tych 7 wiadomości i dla pozostałych 53. Jeśli na „wspólnych wątkach” model radzi sobie wyraźnie lepiej, w ocenie trzeba podawać wynik bez nich albo z wyraźnym zastrzeżeniem.

**Czy oceny modelu są wiarygodne i czy próg wybrany w walidacji krzyżowej przenosi się na model końcowy?** Przy 180 przykładach i regularyzacji oceny regresji mogą być ściśnięte wokół 1/3 albo zbyt skrajne. Próg wybierany jest na modelach uczonych na 144 wiadomościach, a model końcowy uczy się na 180. Sprawdzenie na train: pogrupować oceny z walidacji krzyżowej w kilka przedziałów, porównać z rzeczywistym udziałem spamu w każdym z nich i zobaczyć, jak bardzo wybrany próg różni się między przebiegami. Jeśli rozrzut jest duży, próg trzeba traktować jako przybliżony i podawać wyniki dla kilku sąsiednich progów.

_(kolejne niewiadome do uzupełnienia)_
