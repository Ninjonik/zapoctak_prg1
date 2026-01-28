# Dokumentácia k zápočtovému programu: Maticová kalkulačka

Peter Zaťko  
Zimný semester 2025/26  
Programovanie 1

---

## 1. Úvod

Tento dokument slúži ako sprievodná dokumentácia k zápočtovému programu zameranému na implementáciu triedy `Matrix`. Cieľom programu je poskytnúť jednoduchý nástroj na prácu s maticami, ktorý podporuje základné maticové operácie a riešenie sústav lineárnych rovníc.

### 1.1. Zadanie

Jadrom celého programu je jedna trieda `Matrix`, ktorá reprezentuje maticu a umožňuje na nej vykonávať množstvo základných operácií. 
Medzi podporované funkcionality patrí vytváranie matíc, 
aritmetické operácie (sčítanie, odčítanie, násobenie),
transpozícia, hľadanie inverznej matice a riešenie sústav lineárnych rovníc pomocou 
Gaussovej a Gauss-Jordanovej eliminácie.

### 1.2. Presné zadanie

Program umožňuje:

**Vytváranie matíc:**
- Z 2D poľa (zoznamu) `list[list[Number]]` pomocou konštruktora `Matrix(data)`
- Nulová matica pomocou `Matrix.zeros(m, n)`
- Jednotková matica pomocou `Matrix.identity(n)`
- Kópia existujúcej matice pomocou `Matrix.from_matrix(other)` alebo `matrix.copy()`

**Prístup k dátam:**
- **Indexovanie od 1** (nie od 0!) – `A[i]` vracia i-ty riadok ako `tuple`, `A[i, j]` vracia prvok na i-tom riadku v j-tom stĺpci
- Iterovanie ako v 2D poli: `for row in matrix: ...`
- Vlastnosti matice: `.rows`, `.cols`, `.shape`, `.empty`

**Aritmetické operácie (vracajú nový objekt):**
- **Sčítanie matíc** – `A + B` (matice musia mať rovnaký rozmer)
  - Implementované pomocou `__add__` a `__radd__`
  - Funguje symetricky z oboch strán, je komutatívne
- **Odčítanie matíc** – `A - B` (matice musia mať rovnaký rozmer)
  - Implementované pomocou `__sub__`
  - Nie je komutatívny
- **Násobenie skalárom** – `A * skalár` aj `skalár * A`
  - Implementované pomocou `__mul__` a `__rmul__`
  - Funguje symetricky z oboch strán, je komutatívne
- **Maticový súčin** – `A @ B`
  - Implementované pomocou `__matmul__`
  - Počet stĺpcov `A` musí byť rovný počtu riadkov `B`
  - Nie je komutatívny

**Transpozícia (in-place):**
- `transpose()` – transponuje maticu na mieste, mení `.rows` a `.cols`
- Pre zachovanie originálu: `B = A.copy(); B.transpose()`

**Gaussova a Gauss-Jordanova eliminácia (in-place):**
- `ref()` – Gaussova eliminácia, prevedie maticu do REF (row echelon form)
- `rref()` – Gauss-Jordanova eliminácia, prevedie maticu do RREF (reduced row echelon form)

**Charakteristiky matice:**
- `.is_square` – či je matica štvorcová
- `.is_regular` – či je matica regulárna
- `.is_symmetric` – či je matica symetrická
- `rank()` – hodnosť matice (vykonáva REF, časovo netriviálne)

**Inverzia matice:**
- `invert()` – vracia novú maticu (inverznú)
- Funguje len pre regulárne štvorcové matice

**Riešenie sústavy lineárnych rovníc:**
- `get_solutions()` – vstup: rozšírená matica `(A|b)`
- Vracia: žiadne riešenie, jedno riešenie, nekonečne veľa riešení vyjadrené parametricky

**Pomocné operácie:**
- `swap_rows(i, j)` – výmena riadkov (in-place, vracia `self` pre reťazenie)
- `is_row_empty(row)` – test, či je riadok nulový
- `__eq__` – porovnanie matíc
---

## 2. Návrh a implementácia

### 2.1. Zvolené algoritmy a dátové štruktúry

**Dátová štruktúra matice:**
Hlavnou dátovou štruktúrou je trieda `Matrix`. 
Samotné dáta matice sú uložené ako 2D zoznam (zoznam zoznamov), 
kde každý vnútorný zoznam reprezentuje jeden riadok matice. 

**Kľúčové algoritmy:**

*   **Maticový súčin:** Použitý bol štandardný algoritmus s časovou zložitosťou O(n³).
*   **Výpočet inverznej matice:** Výpočet je založený na **Gaussovej-Jordanovej eliminácii**, ktorá redukuje maticu do RREF.
    *   Pri hľadaní **inverznej matice** sa pôvodná matica rozšíri o jednotkovú maticu. Aplikovaním Gaussovej-Jordanovej eliminácie sa pôvodná matica transformuje na jednotkovú a z pôvodnej jednotkovej matice tak vznikne napravo inverzná matica a naľavo jednotková matica.
*   **Riešenie sústav lineárnych rovníc:** Rovnako sa využíva Gauss-Jordanova eliminácia na prevedenie rozšírenej matice sústavy do REF, z ktorého sa následne spätnou substitúciou dopočítajú neznáme.

### 2.2. Štruktúra programu a operácie

Program je rozdelený do dvoch hlavných súborov: `main.py` a `testovacia_sada.py`.

*   **`main.py`**: Obsahuje definíciu triedy `Matrix` a všetkých jej metód. Taktiež implementuje interaktívne užívateľské menu pre prácu s programom.
*   **`testovacia_sada.py`**: Slúži na automatizované testovanie funkčnosti triedy `Matrix` a jej metód.

**Prehľad operácií a správania metód:**

Väčšina operácií v triede `Matrix` je navrhnutá tak, aby **vytvárala a vracala nový objekt `Matrix`** a neupravovala pôvodné objekty. Týmto sa predchádza neočakávaným side effectom.

*   **Aritmetické operácie (`+`, `-`, `*`, `**`)**: Vždy vracajú novú maticu ako výsledok.
*   **`transponuj()`**: Vytvorí a vráti novú, transponovanú maticu.
*   **`inverzia()`**: Vráti novú, inverznú maticu.
*   **`ries_sustavu()`**: Vráti riešenie sústavy ako nový objekt `Matrix` (vektor riešení).

**Výnimku** tvoria interné pomocné metódy používané napríklad pri Gaussovej eliminácii, ktoré pracujú priamo s dátami matice ("in-place") pre zvýšenie efektivity.

---

## 3. Používateľská príručka

### 3.1. Spustenie programu

Program sa spúšťa z príkazového riadku pomocou interpretu jazyka Python:

```bash
python3 main.py
```

Po spustení sa zobrazí hlavné menu, ktoré ponúka nasledujúce možnosti:

1.  **Práca s maticami:** Vstup do podmenu pre definovanie a správu matíc.
2.  **Operácie s maticami:** Vstup do podmenu pre vykonávanie maticových operácií.
3.  **Koniec:** Ukončenie programu.

### 3.2. Príprava vstupných dát

Program umožňuje načítať maticu zo súboru. Formát súboru musí byť nasledovný:

*   Každý riadok súboru predstavuje jeden riadok matice.
*   Čísla v riadku sú oddelené medzerou.
*   Všetky riadky musia mať rovnaký počet čísel.

**Príklad súboru `matica.txt`:**
```
1 2 3
4 5 6
7 8 9
```

### 3.3. Interpretácia výstupných dát

Výsledky operácií (matice) sa zobrazujú priamo na obrazovku v prehľadnom formáte. Program sa snaží výstup formátovať tak, aby bol ľahko čitateľný. Napríklad, pri riešení sústavy rovníc program vypíše výsledný vektor neznámych.

---

## 4. Zhodnotenie a záver

### 4.1. Priebeh prác a nadobudnuté skúsenosti

Práca na programe prebiehala v niekoľkých fázach. Najprv bola navrhnutá základná štruktúra triedy `Matrix` a implementované jednoduché operácie ako sčítanie a násobenie skalárom. Najväčšou výzvou bola správna a robustná implementácia Gaussovej-Jordanovej eliminácie, ktorá je kľúčová pre výpočet determinantu, inverzie a riešenie sústav rovníc. Počas implementácie sa ukázalo ako dôležité dôkladne ošetrovať špeciálne prípady, ako sú singulárne matice alebo nekompatibilné rozmery matíc pri operáciách.

Zaujímavým zistením bolo, ako úzko sú prepojené zdanlivo odlišné operácie. Implementácia jednej centrálnej metódy (Gaussova eliminácia) umožnila elegantne vyriešiť viacero problémov naraz.

### 4.2. Možné rozšírenia

Hoci program spĺňa všetky požiadavky zadania, existuje niekoľko možností na jeho ďalšie rozšírenie:

*   **Optimalizácia výkonu:** Pre prácu s veľmi veľkými maticami by bolo možné prepísať kritické časti kódu s využitím knižnice `NumPy`, ktorá je implementovaná v jazyku C a je výrazne rýchlejšia.
*   **Rozšírenie o ďalšie operácie:** Program by sa dal doplniť o ďalšie pokročilé funkcie z lineárnej algebry, ako je výpočet vlastných čísel a vlastných vektorov, alebo implementácia rôznych typov maticových rozkladov (napr. LU, QR).
*   **Grafické užívateľské rozhranie (GUI):** Pre pohodlnejšiu prácu by bolo možné vytvoriť grafické rozhranie, ktoré by umožnilo vizuálne zadávať matice a zobrazovať výsledky.

### 4.3. Záver

Práca na tomto projekte bola cennou skúsenosťou v oblasti objektovo orientovaného programovania v Pythone a zároveň skvelým spôsobom, ako si precvičiť a lepšie pochopiť koncepty z lineárnej algebry. Výsledkom je funkčný a solídne navrhnutý program, ktorý slúži ako dobrý základ pre prípadné ďalšie rozšírenia. Rozhodnutie implementovať algoritmy od základu namiesto použitia hotových knižníc bolo kľúčové pre hlbšie pochopenie ich fungovania.

---

## 5. Sada testovacích príkladov

Súčasťou projektu je súbor `testovacia_sada.py`, ktorý obsahuje sadu testov pre overenie správnosti implementovaných metód. Testy pokrývajú:

*   Vytváranie matíc.
*   Základné aritmetické operácie.
*   Výpočet determinantu pre regulárne aj singulárne matice.
*   Výpočet inverznej matice.
*   Riešenie sústav lineárnych rovníc s jedným, žiadnym aj nekonečným počtom riešení.

Spustenie testov poskytuje rýchlu spätnú väzbu o funkčnosti programu a pomáha odhaliť prípadné chyby pri úpravách kódu.
