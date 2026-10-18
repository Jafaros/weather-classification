# Předpověď deště – weatherAUS

V tomto cvičení porovnávám Gaussian naive Bayes a kNN na datasetu weatherAUS.
Cílem je podle dnešního počasí odhadnout, jestli bude zítra pršet.
U kNN používám 5 nejbližších sousedů a výsledky obou metod porovnávám pomocí
10-fold cross validace.

## Spuštění

```sh
python main.py
```

Program je potřeba spustit z hlavní složky projektu a dataset mít v
`data/weatherAUS.csv`. Do konzole se vypíšou výsledky a otevře se obrázek
s confusion matrix pro oba modely. Výsledky se také uloží do složky `results`:

- `confusion_matrices.png` – matice pro oba modely.
- `GaussianNB_fold_metrics.csv`, `kNN_fold_metrics.csv` – výsledky jednotlivých foldů.

## Výběr a příprava dat

Z datasetu jsem vybral tyto sloupce:

- `MinTemp`, `MaxTemp` – nejnižší a nejvyšší teplota.
- `Rainfall` – množství dnešních srážek.
- `WindGustSpeed`, `WindGustDir` – rychlost a směr nárazů větru.
- `Humidity9am`, `Humidity3pm` – ranní a odpolední vlhkost.
- `Pressure9am`, `Pressure3pm` – ranní a odpolední tlak.
- `Date` – pro zachování informace o ročním období.

Sloupec `RainTomorrow` je výsledek, který se snažím předpovědět.
Hodnoty `No` a `Yes` převádím na 0 a 1.

Datum převádím na pořadí dne v roce a potom na sinus a kosinus podle délky roku.
Směr větru převádím nejprve na stupně (N = 0°, E = 90° atd.) a také na sinus
a kosinus. Díky tomu jsou si blízké i hodnoty kolem přelomu roku nebo směry
kolem severu. Celkem tak vznikne 12 číselných vstupů.

`RainToday` jsem vynechal, protože podobnou informaci už obsahuje `Rainfall`.
Další měření teploty a větru jsem pro tuto verzi nepoužil.
`Sunshine`, `Evaporation` a oblačnost mají hodně chybějících hodnot.
Nepoužívám ani `Location`, takže modely zpracovávají všechny stanice společně
bez rozlišení místa. Tento výběr sloupců je výchozí varianta, jiné kombinace
jsem zatím neporovnával.

Důležité je vyřadit `RISK_MM`, protože obsahuje množství zítřejších srážek.
Model by tak dostal informaci, kterou má teprve předpovědět.

Řádky s NA nebo neplatnou hodnotou ve vybraných sloupcích odstraňuji.
Chybějící hodnoty v nepoužitých sloupcích nevadí. Z původních 142 193 řádků
zůstalo 119 674, z toho 93 464 bez deště a 26 210 s deštěm.
Původní CSV se nemění.

U kNN ještě používám `StandardScaler`, aby například tlak kvůli svým větším
číselným hodnotám nepřevážil ostatní údaje. Počítá se vždy jen z trénovacích
dat daného foldu, aby se do trénování nedostaly informace z testovací části.
U GaussianNB tento krok není potřeba.

## Vyhodnocení

Data rozděluji pomocí `StratifiedKFold` na 10 částí. V každé je přibližně stejný
poměr deštivých a suchých dnů jako v celém zpracovaném datasetu. Model se vždy
naučí na devíti částech a otestuje na zbývající. To se opakuje desetkrát,
takže každý řádek jednou slouží jako testovací.

Oba modely dostávají stejné rozdělení. Používám `shuffle=True` a
`random_state=42`, aby šel běh zopakovat. U kNN počítám eukleidovskou vzdálenost
a všech 5 sousedů má stejnou váhu. Jiné hodnoty k jsem zatím netestoval.

Program vypisuje průměr a směrodatnou odchylku výsledků přes všech 10 foldů:

- **Accuracy** – podíl všech správných předpovědí.
- **Balanced accuracy** – průměr úspěšnosti pro suché a deštivé dny.
- **Precision pro Yes** – jak často má model pravdu, když předpoví déšť.
- **Recall pro Yes** – kolik skutečně deštivých dnů model zachytí.
- **F1 pro Yes** – společné hodnocení precision a recall.

Confusion matrix obsahuje výsledky ze všech testovacích částí dohromady.
Řádky ukazují skutečné hodnoty a sloupce předpovědi, vždy v pořadí `No`, `Yes`.

### Naměřené výsledky

Na přiložených datech vyšly tyto průměrné výsledky:

| Model       | Accuracy | Balanced accuracy | Precision Yes | Recall Yes | F1 Yes |
| ----------- | -------: | ----------------: | ------------: | ---------: | -----: |
| GaussianNB  |   0,8314 |            0,6970 |        0,6678 |     0,4578 | 0,5432 |
| kNN (k = 5) |   0,8424 |            0,7248 |        0,6871 |     0,5154 | 0,5889 |

O něco lépe vyšel kNN. Accuracy kolem 84 % vypadá dobře, ale samotné číslo
trochu klame: kdyby model pokaždé řekl, že pršet nebude, měl by už 78,1 %.
Suchých dnů je totiž v datech mnohem víc.

Když kNN předpoví déšť, má pravdu přibližně v 69 % případů. Zachytí ale jen
asi 52 % všech deštivých dnů, takže skoro polovinu přehlédne.
Předpovídat déšť je tedy pro oba modely slabší stránka.

Výsledky platí pro řádky, které zůstaly po odstranění NA. Protože jsou data
rozdělená náhodně, neověřoval jsem zvlášť úspěšnost na budoucích obdobích ani
na nových stanicích. Jako další pokus by šlo porovnat jiné k, přidat další
sloupce nebo vyzkoušet model jen pro jedno místo.

![Confusion matrix obou modelů](results/confusion_matrices.png)
