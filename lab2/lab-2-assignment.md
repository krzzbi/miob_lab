# Laboratorium 2. Od problemu do planu rozwiązania

<span id="cel-zadania"></span>

## Problem i cel

Zespół chce przygotować lokalny prototyp rozpoznający spam na podstawie tekstu wiadomości. Otrzymujesz dane pochodzące z **Apache SpamAssassin**, zawierające wiadomości z etykietami `spam` i `ham` — prawidłowa poczta — oraz przygotowany podział na dane do uczenia i oceny.

**Z pomocą agenta poznaj problem, wybierz podejście i przygotuj uzasadniony plan budowy oraz oceny prototypu.** Nie musisz wcześniej znać uczenia maszynowego. Agent może wyjaśniać pojęcia, analizować dane i proponować rozwiązania. Twoim zadaniem jest zrozumieć i uzasadnić wybory zawarte w planie.

<span id="granice"></span>

Prototyp ma działać lokalnie, bez podłączania do skrzynki i bez usuwania wiadomości. Zachowaj oryginalne dane i przygotowany podział. Ewentualne przekształcenia opisz w planie. **Laboratorium kończy się planem — nie wymaga implementacji całego rozwiązania ani wytrenowania końcowego modelu.**

**Od czego zacząć?**

Przed rozpoczęciem skorzystaj z instrukcji [Przygotowanie do pracy](https://agents-ml-course-2026.czuba-przemyslaw.chatgpt.site/przygotowanie). Pobierz ZIP z danymi i rozpakuj go. Otwórz folder `project/` w aplikacji agenta. Przeczytaj [opis danych w README](https://agents-ml-course-2026.czuba-przemyslaw.chatgpt.site/lab-2-data) i obejrzyj z agentem zawartość plików CSV. W `data/` znajdują się `train.csv` do uczenia i `validation.csv` do oceny. Zapoznaj się z kolumnami `text`, `label` i `message_id`. Zbiór jest historycznym podzbiorem dydaktycznym. Wynik na nim nie potwierdzi skuteczności na współczesnej poczcie.

**Podczas tego laboratorium zapoznaj się z dostarczonymi danymi, wyjaśnij potrzebne pojęcia i wybierz podejście.** Uczenie modelu i pomiar jego jakości opisz jako przyszłe kroki. Małą próbę wykonaj tylko wtedy, gdy pomoże podjąć decyzję. Paczka zawiera wyłącznie dane i ich dokumentację. Nie narzuca modelu ani sposobu reprezentacji tekstu.

<span id="plan-eksperymentu"></span>

## Co trzeba zrozumieć i rozstrzygnąć?

Poniższa tabela wskazuje obszary potrzebne do przygotowania planu. Nie wszystkie wymagają swobodnego wyboru: część dotyczy zrozumienia dostarczonych danych i ograniczeń zadania.

**Jeżeli pytanie zawiera nieznane pojęcie, poproś agenta o wyjaśnienie na przykładzie z danych. Następnie ustal, jak ta wiedza wpływa na Twój plan.** Kolejność rozmowy jest dowolna. Wcześniejsze decyzje możesz zmieniać po zdobyciu nowych informacji.

| Obszar planu | Co trzeba zrozumieć i rozstrzygnąć? |
|---|---|
| **1. Dane i ich przygotowanie** | Co opisuje pojedynczy przykład? Czym jest etykieta i do czego służy? Sprawdź dostępne informacje, braki i powtórzenia. Zdecyduj, jakie przygotowanie danych jest potrzebne i dlaczego. |
| **2. Reprezentacja wiadomości** | W jakiej postaci model może otrzymać wiadomość? Wskaż metodę reprezentacji tekstu i informacje, które wykorzystasz. Wyjaśnij, co taka reprezentacja zachowuje, a co pomija. |
| **3. Wybór modelu** | Czym jest model i jak uczy się na przykładach? Wskaż rodzaj modelu pasujący do wybranej reprezentacji i celu. Sam zapis „zastosować klasyfikator” nie określa jeszcze wyboru. Porównaj wybrany model z jedną sensowną alternatywą, uwzględniając wymagania i ograniczenia. |
| **4. Uczenie i późniejsze użycie** | Czym różni się uczenie od klasyfikowania nowej wiadomości? Ustal, kiedy potrzebne są etykiety, co powstaje podczas uczenia i co trzeba zachować do późniejszego użycia. |
| **5. Ocena rozwiązania** | Dlaczego dane do oceny pozostają poza uczeniem? Wybierz miary oceny i wyjaśnij, co pokazują. Wyjaśnij, jak ocena uwzględni przeoczony spam oraz prawidłowe wiadomości błędnie oznaczone jako spam. |
| **6. Niewiadome i dalsze sprawdzenia** | Które założenia mają podstawę w danych lub dokumentacji, a które wymagają sprawdzenia? Ustal, jak je sprawdzić i jaki wynik mógłby skłonić Cię do zmiany podejścia. |

## Jak pracować z agentem?

Obejrzyj rzeczywiste dane. Proś o wyjaśnienia, przykłady i propozycje potrzebne do podjęcia decyzji. Istotne twierdzenia dotyczące projektu sprawdzaj w danych lub dokumentacji — samo przekonujące wyjaśnienie agenta nie potwierdza ich poprawności.

Możesz wykonać małą próbę, jeśli pomoże rozstrzygnąć konkretną niewiadomą. Przed jej wykonaniem ustal, czego chcesz się dowiedzieć i jak wynik wpłynie na plan. Próby wykonuj osobno, bez nadpisywania dostarczonych danych.

Nie musisz samodzielnie wymyślać rozwiązania ani odrzucać poprawnej propozycji. **Przyjęcie propozycji agenta również jest decyzją — powinieneś umieć wyjaśnić, dlaczego ją przyjmujesz.** W razie trudności skorzystaj z [Cheatsheetu](https://agents-ml-course-2026.czuba-przemyslaw.chatgpt.site/handout).

<span id="sprawozdanie"></span>

## Co powinno zawierać sprawozdanie?

Przygotuj jeden krótki dokument zawierający plan i jego uzasadnienie:

1. **Cel i zakres.** Co prototyp otrzymuje, co zwraca i do czego ma służyć?
2. **Wybrane podejście i uzasadnienie.** Opisz potrzebne przygotowanie danych, metodę reprezentacji tekstu i rodzaj modelu. Wyjaśnij, dlaczego wybrano to podejście. Porównaj wybrany model z jedną alternatywą. Przy najważniejszych wyborach wskaż obserwację danych, informację z dokumentacji lub porównanie, które przemówiło za decyzją.
3. **Kroki planu i ich zależności.** Prześledź drogę od dostarczonych wiadomości do oceny odpowiedzi modelu. Wskaż kolejność czynności, potrzebne dane i wyniki, które trzeba zachować. Wyjaśnij, co powstaje w każdym kroku i do czego jest potrzebne dalej.
4. **Planowana ocena.** Wskaż dane do oceny, sposób porównania przewidywań z etykietami oraz wybrane miary. Wyjaśnij, co miary pokażą o przeoczonym spamie i prawidłowych wiadomościach uznanych za spam.
5. **Niewiadome i możliwe korekty.** Wskaż, które założenia wymagają jeszcze sprawdzenia. Opisz, jak je sprawdzisz i co należałoby zmienić w planie, gdyby okazały się błędne.

Plan może pozostać na poziomie koncepcji. Nie trzeba ustalać wszystkich ustawień, wybierać funkcji biblioteki ani podawać kodu i komend. Wyjaśnienia powinny dotyczyć wybranego rozwiązania. Nie przygotowuj osobnego słownika definicji ML.

**Przed oddaniem sprawdź własne rozumienie.** Spróbuj bez pomocy agenta wyjaśnić, jak plan prowadzi do celu, dlaczego wybrano dane podejście i co może wymagać zmiany podczas wykonania. Następnie możesz poprosić agenta o sprawdzenie rozumowania. Jeśli zauważysz błąd, popraw plan lub wyjaśnienia bezpośrednio w odpowiednich częściach sprawozdania. Nie trzeba dopisywać osobnych odpowiedzi ani powtarzać tych samych informacji.

<div class="discussion-question">

**Kiedy plan jest wystarczający?**

**Plan nie musi być idealny ani opisywać wszystkich szczegółów implementacji.** Powinien określać cel, wybrane podejście, główne kroki, potrzebne dane i sposób oceny wyniku. Kroki muszą do siebie pasować: jeśli któryś wymaga przygotowanego modelu lub przetworzonych danych, plan powinien wskazywać, skąd się wezmą.

Część szczegółów i problemów ujawni się dopiero podczas wykonania. Zapisz istotne niewiadome i sposób ich sprawdzenia. Jeśli odpowiedź może zmienić wybór rozwiązania, uwzględnij sprawdzenie przed pracą, która od tego wyboru zależy. Plan można później poprawić na podstawie nowych informacji.

Zakończ pracę, gdy plan spełnia te warunki i potrafisz uzasadnić jego najważniejsze decyzje. Nie musisz dowodzić, że wybrany model osiągnie dobrą jakość.

</div>

<span id="oddanie"></span>

## Oddanie i ocena

Pracę zapisz w folderze `lab2/` w swoim repozytorium na GitHubie, wspólnym dla całego semestru. Plan z uzasadnieniem umieść w pliku `sprawozdanie.md`. Na początku dodaj link do repozytorium. Nie jest wymagany osobny raport z wykonania eksperymentu.

Podaj także aplikację agenta, nazwę używanego modelu językowego i poziom rozumowania, jeśli aplikacja go udostępnia. Jeśli zmieniano te ustawienia, wymień użyte warianty. To informacja diagnostyczna, nie element punktowany.

W podfolderze `sesje/` dołącz pełne zapisy wszystkich sesji wykorzystanych podczas zadania, z nazwami wskazującymi kolejność i w formacie eksportu aplikacji. **Zapis sesji jest wymagany, ale jego treść nie podlega ocenie ani punktacji.** Służy prowadzącemu do rozpoznawania trudności i poprawy zajęć. Zachowaj pytania o podstawy i nieudane próby.

**Nie ma jednego oczekiwanego rozwiązania.** Różne plany mogą być poprawne, jeśli odpowiadają celowi, pasują do dostępnych danych, są wykonalne na poziomie głównych kroków i pozwalają sensownie ocenić rezultat.

Oceniane będą spójność planu, uzasadnienie wyborów i rozumienie ich konsekwencji. Podczas krótkich pytań lub wejściówki bez agenta wyjaśnisz najważniejsze decyzje i wskażesz, co wymagałoby zmiany po zmianie warunków zadania. Nie trzeba pamiętać składni poleceń ani wzorów modelu. Długość planu i liczba pytań zadanych agentowi nie są kryteriami oceny.
