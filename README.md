# Předpověď deště – weatherAUS

V tomto cvičení porovnávám Gaussian naive Bayes a kNN na datasetu weatherAUS.
Cílem je podle dnešního počasí odhadnout, jestli bude zítra pršet.
U kNN používám 5 nejbližších sousedů a výsledky obou metod porovnávám pomocí 10-fold cross validace.

## Spuštění

```sh
python main.py
```

Program je potřeba spustit z hlavní složky projektu a dataset mít v
`data/weatherAUS.csv`. Do konzole se vypíšou výsledky a otevře se obrázek
s confusion matrix pro oba modely. Výsledky se také uloží do složky `results`:

- `confusion_matrices.png` – matice pro oba modely
- `GaussianNB_fold_metrics.csv`, `kNN_fold_metrics.csv` – výsledky jednotlivých foldů

## Výběr a příprava dat

Z datasetu jsem vybral tyto sloupce:

- `MinTemp`, `MaxTemp` – nejnižší a nejvyšší teplota
- `Rainfall` – množství dnešních srážek
- `WindGustSpeed`, `WindGustDir` – rychlost a směr nárazů větru
- `Humidity9am`, `Humidity3pm` – ranní a odpolední vlhkost
- `Pressure9am`, `Pressure3pm` – ranní a odpolední tlak
- `Date` – pro zachování informace o ročním období

Sloupec `RainTomorrow` je výsledek, který se snažím předpovědět.
Hodnoty `No` a `Yes` převádím na 0 a 1.

Datum převádím na pořadí dne v roce a potom na sinus a kosinus podle délky roku.
Směr větru převádím nejprve na stupně (N = 0°, E = 90° atd.) a také na sinus
a kosinus. Díky tomu jsou si blízké i hodnoty kolem přelomu roku nebo směry
kolem severu. Celkem tak vznikne 12 číselných vstupů.

`RainToday` jsem vynechal, protože podobnou informaci už obsahuje `Rainfall`.
`Sunshine`, `Evaporation` a oblačnost mají hodně chybějících hodnot.
Nepoužívám ani `Location`, takže modely zpracovávají všechny stanice společně
bez rozlišení místa.

Důležité je vyřadit `RISK_MM`, protože obsahuje množství zítřejších srážek.
Model by tak dostal informaci, kterou má teprve předpovědět.

Řádky s NA nebo neplatnou hodnotou ve vybraných sloupcích odstraňuji.

U kNN ještě používám `StandardScaler`, aby například tlak kvůli svým větším
číselným hodnotám nepřevážil ostatní údaje. Počítá se vždy jen z trénovacích
dat daného foldu, aby se do trénování nedostaly informace z testovací části.
U GaussianNB tento krok není potřeba.

## Vyhodnocení

Data rozděluji pomocí `StratifiedKFold` na 10 částí. V každé je přibližně stejný
poměr deštivých a suchých dnů jako v celém zpracovaném datasetu.

### Naměřené výsledky

Na přiložených datech vyšly tyto průměrné výsledky:

| Model       | Accuracy | Balanced accuracy | Precision Yes | Recall Yes | F1 Yes |
| ----------- | -------: | ----------------: | ------------: | ---------: | -----: |
| GaussianNB  |   0,8314 |            0,6970 |        0,6678 |     0,4578 | 0,5432 |
| kNN (k = 5) |   0,8424 |            0,7248 |        0,6871 |     0,5154 | 0,5889 |

![Confusion matrix obou modelů](results/confusion_matrices.png)
