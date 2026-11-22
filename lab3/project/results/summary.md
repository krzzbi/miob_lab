# Podsumowanie przebiegu prototypu rozpoznawania spamu

## 1. Wybrany wariant i wyniki na `train.csv` (5-krotna CV, ziarno 2026)

- **Wybrany model:** Regresja logistyczna L2 (`C = 10.0`), próg OOF = `0.328995`, słownik na całym `train.csv` (`min_df=2`): `2632` tokenów.
- **Wynik CV (OOF, 180 wiadomości: 120 ham / 60 spam):**
  - TP = `58/60` (czułość `96,7%`), FP = `3/120` (FPR `2,5%`), precyzja `58/61` (`95,1%`), PR-AUC = `0.9910`, dokładność `175/180` (`97,2%`).
  - Rozrzut własnego progu między foldami (`max_fp=0`, ≤ 2,5%): `0.3472 – 0.5301` (rozpiętość `0.1829`).
- **Model alternatywny (informacyjnie):** Wielomianowy Naiwny Bayes (`α = 0.1`), próg OOF = `0.580394` — CV: TP = `57/60` (`95,0%`), FP = `3/120` (`2,5%`), PR-AUC = `0.9943`, rozrzut progu `0.2234`.
- **Zamrożenie modeli przed otwarciem `validation.csv`:**
  - `models/spam_classifier.json`: `68cc96ef843e24f4b2d78fd79d6a49ea06fd4d0811da4a2488afe2fa68a3a28c`
  - `models/alternative_model.json`: `41251e0a386590884eefa760db4c800243c29fff667a2f70d50e542b2dd08763`

## 2. Jednorazowa ocena na `validation.csv` (60 wiadomości: 40 ham / 20 spam)

| Wariant | TP / FP / FN / TN | FPR (fałszywe alarmy) | Czułość (spam) | Precyzja (spam) | PR-AUC | Dokładność |
|---|---|---|---|---|---|---|
| **Regresja L2 (`C=10.0`, próg `0.3290`)** | **19 / 0 / 1 / 40** | **0/40 (0,0%)** | **19/20 (95,0%)** | **19/19 (100,0%)** | **1.0000** | **59/60 (98,3%)** |
| Model „zawsze ham” | 0 / 0 / 20 / 40 | 0/40 (0,0%) | 0/20 (0,0%) | 0/0 (0,0%) | — | 40/60 (66,7%) |
| Reguła `<MONEY> i remove → spam` | 5 / 1 / 15 / 39 | 1/40 (2,5%) | 5/20 (25,0%) | 5/6 (83,3%) | — | 44/60 (73,3%) |
| Naiwny Bayes (`α=0.1`, próg `0.5804`, inf.) | 19 / 0 / 1 / 40 | 0/40 (0,0%) | 19/20 (95,0%) | 19/19 (100,0%) | 0.9976 | 59/60 (98,3%) |

- **Progi informacyjne wybranego modelu (`0.25`, `0.40`, `0.50`):** dla wszystkich trzech progów wynik jest identyczny (`19/20` TP, `0/40` FP, `59/60` dokładność) — próg w modelu pozostaje niezmieniony (`0.328995`).
- **Jedyny błąd na `validation.csv` (0 FP, 1 FN):** `017f0817d4c92951` (`score = 0.2173`, `Subject: Newsletter Coach'Invest - Septembre 2002`) — francuskojęzyczny newsletter komercyjny; większość słów nie występuje w anglojęzycznym słowniku `train`, kwota `0.34 Euros/min` nie dopasowuje wzorca `<MONEY>`, a tokeny `<NUM>`, `la`, `de`, `re`, `dont` obniżyły ocenę mimo silnego `<CAPS>`.

## 3. Diagnostyka po ocenie (sekcja 5 planu — bez zmian w modelu)

- **(a) Kalibracja OOF (CV na `train`):** monotoniczna — `[0.0, 0.2)`: `1/106` spam (`0,9%`), `[0.2, 0.4)`: `3/15` (`20,0%`), `[0.4, 0.6)`: `10/13` (`76,9%`), `[0.6, 0.8)`: `10/10` (`100,0%`), `[0.8, 1.0]`: `36/36` (`100,0%`).
- **(b) Maskowanie vs brak maskowania:** oba warianty dają w CV `58/60` TP i `3/120` FP (`PR-AUC`: `0.9910` z maskowaniem vs `0.9930` bez maskowania) oraz na `validation` `19/20` TP, `0/40` FP, `PR-AUC = 1.0000`. Usunięcie śladów źródła nie pogorszyło jakości.
- **(c) Perturbacje na kopiach 20 spamów z `validation`:**
  - Dopisanie stopki listy ILUG z `train`: **`0/20` zmian oceny** (`max_abs_diff = 0.0`), `0/20` zmian decyzji — maskowanie usuwa stopkę w całości na nowych danych.
  - Dopisanie nie-maskowanej linii przypisania cytatu (`On Mon, Sep 23, 2002 at 10:15:00AM +0100, David wrote:` + `> Here is the previous message.`): **`0/20` zmian decyzji**, mediana spadku oceny **`0.0577`** (zakres `0.0026 – 0.1812`) — model jest zauważalnie wrażliwy na nie-maskowane ślady stylu cytowań i archiwum z 2002 r.
- **(d) Wspólne wątki (`train` vs `validation`):** dokładnie `7` wspólnych wiadomości (`1` spam, `6` ham) i `53` pozostałe (`19` spam, `34` ham). Wyniki: wspólne `7/7` (`100,0%`, FPR `0/6`, czułość `1/1`), pozostałe `52/53` (`98,1%`, FPR `0/34`, czułość `18/19 = 94,7%`, PR-AUC `1.0000`).

## 4. Lista odstępstw od planu (`plan-lab2.md`)

1. **Własna implementacja w bibliotece standardowej Pythona 3.12 i solver Newtona:** ze względu na brak `numpy`/`scipy`/`scikit-learn`/`joblib` w środowisku i mało wolnego miejsca na dysku (`11 GiB`) zaimplementowano od zera TF-IDF, wielomianowy Naiwny Bayes oraz regresję logistyczną L2 (początkowo L-BFGS, który nie zbiegł w 500 iteracjach na `train`, zastąpiony dokładną metodą Newtona z tożsamością Woodbury'ego i rozkładem Cholesky'ego). Artefakty zapisano w czytelnym JSON.
2. **Spadek liczby spamów z tokenem `remove` (`24 → 17`) wskutek `<URL>`:** samo maskowanie artefaktów źródła zachowuje wszystkie `24/60` spamów zawierających `\bremove\b` (`33/33` dla `remov`, `9/9` dla `unsubscrib`), ale w `7` spamach słowo `remove` występowało wyłącznie w adresach URL (np. `remove.php`, `poker-remove.cgi`), które zgodnie z krokiem 2.2 są zamieniane na `<URL>`, więc w tokenach (zarówno z maskowaniem, jak i bez) pozostaje `17/60`.
3. **Poprawka wzorców maskowania po testach pokrycia i pozostałości:** pierwotny wzorzec banera Yahoo! Groups (`~->` zamiast `~-+>`) miał `0` dopasowań (pozostało `7` banerów), a ogólny wzorzec Mailmana pomijał `/lists/listinfo/` (`Spamassassin-devel`) i jednoliniową stopkę `FoRK` (`2` pozostałości `listinfo`). Oba błędy wykryto testami na `train.csv` i naprawiono przed oceną. Ponadto usuwanie znaczników list (`[ILUG]` itd.) zastosowano w całym tekście (także w cytowanych nagłówkach `Subject:`).
4. **Rozszerzenie maskowania o nagłówki RSS na bramce:** po przeglądzie 30 najsilniejszych wag modelu na `train` (gdzie `#1 date`, `#2 url`, `#9 supplied` były skrótami z bramek RSS `newsisfree`/`boingboing`/`scriptingnews`) dodano za zgodą wzorce `rss_url_header` (`33` ham / `0` spam) i `rss_date_header` (`35` ham / `0` spam, w tym format ISO-8601 i cytowane `> Date:`) oraz powtórzono Część 3 przed oceną na `validation`.
5. **Zmiana jednej asercji testowej w Części 1 przed konsultacją:** w [test_preprocess_tfidf.py](../tests/test_preprocess_tfidf.py) po pierwszym nieudanym przebiegu (`17 != 24`) zmieniono drugą asercję z porównania tokenów z surowym tekstem na porównanie tokenów z maskowaniem vs bez maskowania (`17 == 17`, przy zachowaniu nienaruszonej asercji `24 == 24` na tekście) bez uprzedniej konsultacji; merytorycznie zaakceptowane przez użytkownika przy punkcie zatrzymania po Części 2.
6. **Rozbicie sekcji 5 na bramkę przed oceną i diagnozę po ocenie:** w celu ochrony jednorazowej oceny na `validation.csv` przegląd wag na `train` wykonano przed oceną jako bramkę, a pozostałe diagnozy (b–d) wyłącznie po ocenie bez modyfikowania modelu.

## 5. Ograniczenia rezultatu i niewykonane sprawdzenia

- **Ograniczenia danych i modelu:** zbiór jest mały (`180` / `60`), historyczny (2002 r.), o sztucznej proporcji klas `2:1` i zdominowany przez anglojęzyczne listy techniczne (`easy_ham`). Mimo usunięcia nagłówków i stopek w wagach `ham` pozostają cechy korpusowe/językowe (`re`, `spam`, `wrote`, `cvs`, `rpm`, `exmh`, `linux`, `david`, `sep` w liniach cytowań), co potwierdza spadek ocen spamu o medianę `0.0577` po dopisaniu linii cytatu. Model słabo radzi sobie ze spamem nieanglojęzycznym (jedyny FN to newsletter francuskojęzyczny) i walutą `Euros` (wzorzec `<MONEY>` obejmuje `EUR`, `$`, `£`, `€`, `dollars`, `cents`, ale nie słowo `Euros`).
- **Niewykonane sprawdzenia:**
  - Nie sprawdzono skuteczności na współczesnej poczcie, na wieloczęściowych wiadomościach HTML ani w innym języku niż angielski (poza pojedynczym przypadkiem FN w `validation.csv`).
  - Nie sprawdzono zewnętrznej kalibracji prawdopodobieństw na dużym niezależnym zbiorze (przy 60 wiadomościach `validation` każdy błąd zmienia czułość o 5 pp, a FPR o 2,5 pp).
  - Nie testowano wariantu z maskowaniem linii przypisania cytatu (`On ..., X wrote:`) ani rozszerzenia `<MONEY>` o `Euros` — wnioski te odłożono jako kandydaty do ewentualnej kolejnej wersji, bez modyfikowania ocenionego modelu.
