# Maticová kalkulačka (knižnica + CLI)

**Autor:** Peter Zaťko  
**Predmet:** Programování 1 (NPRG030)  
**Akademický rok:** 2025/2026, zimný semester  
**Verzia:** 1.0  

---

## Stručné zadanie (anotácia)

Projekt implementuje triedu `Matrix` pre základné operácie lineárnej algebry (tvar/rozmery, sčítanie, odčítanie, skalárny násobok, maticový súčin, transpozícia, REF/RREF, inverzia, riešenie lineárnych sústav v tvare rozšírenej matice).  
Súčasťou je aj jednoduché CLI nad touto knižnicou (bez vlastného “input layeru”), t. j. ovládanie je riešené cez štandardné argumenty príkazového riadku a/alebo priamo cez Python.

---

## Presné zadanie

1. Implementovať dátovú štruktúru pre maticu reálnych čísel (v praxi `int/float`).
2. Podporiť:
   - vytváranie matíc (zoznam, nulová, identita, kópia),
   - prístup k prvkom a riadkom,
   - aritmetiku: `+`, `-`, skalárny násobok, maticový súčin,
   - elementárne riadkové operácie (swap/scale/add),
   - prevod do REF a RREF,
   - výpočet hodnosti,
   - inverziu regulárnej štvorcovej matice,
   - riešenie lineárnej sústavy zo zadanej rozšírenej matice `[A|b]`.
3. Program musí byť použiteľný ako knižnica a mať spustiteľné CLI rozhranie s nápovedou.

---

## Zvolený algoritmus (a prečo)

### REF/RREF
Použitá je Gaussova eliminácia/ Gauss–Jordan eliminácia:
- REF: normalizácia pivotu (škálovanie riadku) + eliminácia prvkov **pod** pivotom.
- RREF: najprv REF, následne eliminácia prvkov **nad** pivotom.

Z praktických dôvodov sa počas eliminácie odstraňujú nulové riadky a riadky sa zoradia podľa pozície pivotu.

### Inverzia
Inverzia sa realizuje štandardne cez rozšírenú maticu `[A|I]` a následnú RREF. Pravá polovica výsledku je potom `A^{-1}`.

### Diskusia alternatív
- Alternatívou by bolo použiť `numpy` (okamžite robustné a rýchle), ale cieľom projektu je vlastná implementácia algoritmov.
- Riadkovú elimináciu by bolo možné robiť s pivotovaním (kvôli numerickej stabilite). V tejto verzii je implementácia jednoduchšia; presnosť je diskutovaná v obmedzeniach.

---

## Štruktúra programu

- `main.py`:
  - trieda `Matrix` (knižnica),
  - CLI (`python main.py --help`) postavené na `argparse`,
  - REPL režim (`python main.py repl`) je len “čistý Python” s importovanou triedou.
- `test_matrix.py`: unit testy porovnávajúce správanie s `numpy` ako referenciou.

---

## Použitie ako knižnica

```python
from main import Matrix

A = Matrix([[1, 2], [3, 4]])
print(A)

print(A.invert())
print((A @ A.invert()))
```

Poznámka: indexovanie je **od 1**:
```python
A[1, 1]  # prvok v 1. riadku a 1. stĺpci
A[2]     # celý 2. riadok ako tuple
```

---

## Použitie ako CLI (pre “užívateľa”)

### Interaktívny režim (QoL príkazy)

Interaktívny prompt funguje ako **normálny Python shell** (môžeš písať výrazy, priradenia, `print(...)`, atď.)
a navyše podporuje pár “command” skráteniek.

Spustenie:
```bash
python main.py
# alebo
python main.py --interactive
```

Príklady Pythonu v promte:
```text
matrix> 2+2
4
matrix> x = 10
matrix> print(x)
10
matrix> A = Matrix([[1,2],[3,4]])
matrix> print(A.invert())
```

Základné príkazy (nadstavba):
- `help` – zobrazí zoznam príkazov
- `set A [[...],[...]]` – nastaví aktuálnu maticu `A` (zároveň nastaví aj Python premennú `A`)
- `show A` – vypíše `A`
- `ref A`, `rref A`, `invert A`, `solve A`
- `py` – prepne do “raw” Python shellu (power-user)
- `exit` – ukončí režim

Príklad:
```text
matrix> set A [[1,2],[3,4]]
matrix> show A
matrix> invert A
```

### Priame CLI príkazy (neinteraktívne)

REF:
```bash
python main.py ref "[[2,1,-1,8],[-3,-1,2,-11],[-2,1,2,-3]]"
```

RREF:
```bash
python main.py rref "[[2,1,-1,8],[-3,-1,2,-11],[-2,1,2,-3]]"
```

Inverzia:
```bash
python main.py invert "[[1,2],[3,4]]"
```

Riešenie sústavy (rozšírená matica `[A|b]`):
```bash
python main.py solve "[[1,2,5],[3,4,11]]"
```

REPL:
```bash
python main.py repl
```

### Formát vstupu (matice v CLI)
Matica sa zadáva ako textový literál **Python 2D listu**, napr.:
- `[[1,2],[3,4]]`
- `[[1,2,5],[3,4,11]]` (rozšírená matica)

Každý riadok musí mať rovnakú dĺžku.

---

## Formát výstupu a interpretácia

- Matice sa vypisujú po riadkoch v zátvorkách.
- `solve` vracia parametrizovaný popis množiny riešení:
  - `No solution.` ak sústava nemá riešenie,
  - `Solution is the entire domain.` pre triviálny prípad (nulová rozšírená matica),
  - inak tvar: `partikulárne riešenie + t0 * v0 + t1 * v1 + ...`

---

## Obmedzenia a známe problémy

1. **Numerická stabilita:** eliminácia nepoužíva pivotovanie; pri “zlých” maticiach môžu vznikať chyby z dôvodu `float` aritmetiky.
2. **Výkon:** implementácia je určená na malé/stredné matice (didaktický cieľ), nie na veľké výpočty.
3. **Indexovanie od 1:** zámerné, ale môže prekvapiť pri Python použití.

---

## Testovacie príklady a testovanie

Testy sú v `test_matrix.py` a používajú `numpy` ako referenciu.

Inštalácia:
```bash
python -m pip install numpy
```

Spustenie:
```bash
python -m unittest test_matrix -v
```

---

## Čo by stálo za doplnenie (možné rozšírenia)

- Čiastočné/úplné pivotovanie v Gaussovej eliminácii.
- Čistejšie API pre riešenia (nevracať `str`, ale štruktúrovaný objekt).
- Parsovanie vstupu aj zo súboru (CSV/TSV) pre CLI.
- Lepšie formátovanie výstupov (napr. zarovnanie stĺpcov).

---

## Záver

Cieľom bolo mať jednoduchú, samostatnú implementáciu základných algoritmov lineárnej algebry v Pythone a zároveň zachovať použiteľnosť projektu ako knižnice aj ako CLI bez budovania vlastnej “mini-shell” vrstvy nad Pythonom.
