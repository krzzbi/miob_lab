# Laboratorium 3. Od planu do działającej implementacji

## Cel i punkt startowy

**Z pomocą agenta wykonaj plan z Labu 2 i przygotuj działający lokalnie prototyp rozpoznawania spamu.** Zorganizuj pracę tak, żeby agent mógł sprawdzać implementację i wykorzystywać wyniki sprawdzeń do dalszego działania.

Punktem startowym są Twój plan i dane z Labu 2. Kod rozwiązania powstaje podczas tego zadania. Zachowaj oryginalne dane oraz podział na uczenie i ocenę. Prototyp nie ma łączyć się ze skrzynką ani usuwać wiadomości.

W razie potrzeby wróć do instrukcji [Przygotowanie do pracy](https://agents-ml-course-2026.czuba-przemyslaw.chatgpt.site/przygotowanie).

## Co trzeba zrobić?

1. **Ustal punkt startowy.** Udostępnij agentowi plan i dane. Sprawdź z nim aktualne pliki, środowisko oraz niewiadome istotne dla rozpoczęcia implementacji.
2. **Dobierz wielkość zleceń.** Możesz poprosić agenta o propozycję podziału pracy. Dla każdej części ustal rezultat, zależności i sposób sprawdzenia. Jeśli wystarczy jedno zlecenie, uzasadnij ten wybór.
3. **Przygotuj delegację i wybierz feedback.** Przed implementacją zapisz główne zlecenie określające rezultat, kontekst, zakres pracy i samodzielności agenta. Ustal, co agent ma sprawdzać, skąd wynika oczekiwany wynik, kiedy sam poprawia kod, kiedy konsultuje problem i po czym rozpoznaje zakończenie zlecenia. Określ też, gdzie mają znaleźć się pliki i podsumowanie. Co najmniej jedno sprawdzenie musi dotyczyć wymaganego zachowania rozwiązania, a nie tylko składni lub stylu kodu.
4. **Zleć wykonanie i obserwuj pracę.** Agent powinien odczytywać wyniki sprawdzeń i na ich podstawie poprawiać implementację, przechodzić dalej albo zgłaszać problem. Jeśli nowe informacje wymagają istotnej zmiany planu lub zlecenia, uzgodnij ją przed pracą, która od niej zależy. Przy podziale na części sprawdźcie również ich współdziałanie.
5. **Uruchom całe rozwiązanie.** Wykonaj z agentem zaplanowany przebieg: przygotowanie danych, uczenie, przewidywanie i wyliczenie wybranych miar. Zachowaj model, potrzebne do jego użycia elementy reprezentacji tekstu oraz przewidywania. Sprawdź, czy po ponownym wczytaniu modelu i tych elementów można przewidywać bez powtarzania uczenia. README projektu powinno określać wymagane zależności, sposób uruchomienia i lokalizacje wyników.

## Co powinno zawierać sprawozdanie?

Przygotuj krótki dokument zawierający:

1. **Delegację** — jeden spójny tekst głównego zlecenia faktycznie przekazanego agentowi przed implementacją. Może odsyłać do planu. Jeśli później go zmieniono, krótko wskaż istotne korekty.
2. **Uzasadnienie organizacji pracy** — dlaczego wybrano takie źródła feedbacku oraz dlaczego zastosowano wskazany podział albo jedno większe zlecenie.
3. **Jeden moment wykorzystania feedbacku** — pokaż konkretny wynik sprawdzenia i wyjaśnij, jakie dalsze działanie z niego wynikało. Może to być także poprawny wynik, po którym agent przeszedł dalej.
4. **Rezultat i ograniczenia** — linki do projektu, README i wyników, najważniejsze odstępstwa od planu oraz informacje o tym, czego nie ukończono lub nie sprawdzono.

<div class="discussion-question">

**Kiedy zadanie jest wykonane?**

Zadanie jest wykonane, gdy zgodnie z README można uruchomić cały zaplanowany przebieg, a projekt zawiera zapisany model gotowy do ponownego użycia, przewidywania dla danych do oceny i wyniki wybranych miar. Sprawozdanie wyjaśnia organizację pracy, wykorzystanie feedbacku i ograniczenia rezultatu. **Nie jest wymagane osiągnięcie określonej jakości klasyfikacji.**

</div>

## Oddanie i ocena

Pracę umieść w folderze `lab3/` w swoim repozytorium na GitHubie. Sprawozdanie zapisz w `sprawozdanie.md`, a na początku podaj link do repozytorium. Dodaj nazwę aplikacji agenta, używanego modelu językowego i poziom rozumowania, jeśli jest dostępny.

W podfolderze `sesje/` dołącz pełne zapisy wykorzystanych sesji, w kolejności pracy i w formacie eksportu aplikacji. **Pełne sesje są wymagane, ale służą wyłącznie diagnozie trudności i nie są punktowane.**

Oceniane będą spójność delegacji z planem, uzasadnienie podziału pracy i wyboru feedbacku, wykorzystanie wyników sprawdzeń oraz działanie i odtwarzalność rozwiązania. Podczas krótkich pytań lub wejściówki bez agenta wyjaśnisz swoje decyzje i znaczenie wybranego wyniku sprawdzenia. Długość zlecenia, liczba podzadań ani liczba poprawek nie są kryteriami oceny.
