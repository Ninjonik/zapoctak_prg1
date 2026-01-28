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

## 2. Ako program používať

### Spustenie

```bash
python main.py
```

Zobrazí sa menu:

```
=== Maticová kalkulačka ===
1. Vytvor maticu
2. Jednotková matica
3. Sčítanie/odčítanie
4. Násobenie (skalár/matica)
5. Transponovať
6. REF/RREF
7. Inverzná matica
8. Riešiť sústavu rovníc
9. Info o matici
0. Koniec
========================
```

### Popis všetkých možností menu

#### 1. Vytvor maticu

Vytvorenie vlastnej matice zadanej zo vstupu.

**Vstup:**
```
Vyber operáciu: 1

Zadaj počet riadkov a stĺpcov (m n):
2 3
Zadaj 2 riadkov po 3 číslach:
1 2 3
4 5 6
```

**Výstup:**
```
1 =
( 1.0 2.0 3.0 )
( 4.0 5.0 6.0 )
```

Matica sa uloží pod číselným menom (1, 2, 3, ...) a dá sa používať v ďalších operáciách.

#### 2. Jednotková matica

Vytvorenie jednotkovej matice zadaného rozmeru.

**Vstup:**
```
Vyber operáciu: 2

Rozmer n: 3
```

**Výstup:**
```
2 =
( 1.0 0.0 0.0 )
( 0.0 1.0 0.0 )
( 0.0 0.0 1.0 )
```

#### 3. Sčítanie/odčítanie

Sčítanie alebo odčítanie dvoch existujúcich matíc.

**Vstup:**
```
Vyber operáciu: 3

Dostupné matice: 1, 2
Prvá matica: 1
Druhá matica: 2
Operácia (+/-): +
```

**Výstup:**
```
3 = 1 + 2
( 2.0 2.0 3.0 )
( 4.0 6.0 6.0 )
```

Matice musia mať rovnaký rozmer, inak program vypíše chybu.

#### 4. Násobenie (skalár/matica)

Násobenie matice skalárom alebo maticový súčin.

**Vstup (skalár):**
```
Vyber operáciu: 4

Dostupné matice: 1, 2, 3
Prvá matica: 1
Násobenie skalárom (s) alebo maticou (m)? s
Skalár: 2.5
```

**Výstup:**
```
4 = 2.5 * 1
( 2.5 5.0 7.5 )
( 10.0 12.5 15.0 )
```

**Vstup (matica):**
```
Vyber operáciu: 4

Dostupné matice: 1, 2, 3, 4
Prvá matica: 2
Násobenie skalárom (s) alebo maticou (m)? m
Druhá matica: 1
```

**Výstup:**
```
5 = 2 @ 1
( ... výsledok maticového súčinu ... )
```

Pri maticovom súčine musí platiť: počet stĺpcov prvej matice = počet riadkov druhej matice.

#### 5. Transponovať

Transpozícia matice (výmena riadkov a stĺpcov).

**Vstup:**
```
Vyber operáciu: 5

Dostupné matice: 1, 2, 3, 4, 5
Matica: 1
```

**Výstup:**
```
6 = 1^T
( 1.0 4.0 )
( 2.0 5.0 )
( 3.0 6.0 )
```

V menu sa vytvára kópia matice, ktorá sa potom transponuje (pôvodná matica zostáva nezmenená).

#### 6. REF/RREF

Gaussova alebo Gauss-Jordanova eliminácia.

**Vstup (REF):**
```
Vyber operáciu: 6

Dostupné matice: 1, 2, 3, 4, 5, 6
Matica: 1
REF (r) alebo RREF (rr)? r
```

**Výstup:**
```
7 = REF(1)
( 1.0 2.0 3.0 )
( 0.0 1.0 2.0 )
```

**Vstup (RREF):**
```
Vyber operáciu: 6

Dostupné matice: 1, 2, 3, 4, 5, 6, 7
Matica: 1
REF (r) alebo RREF (rr)? rr
```

**Výstup:**
```
8 = RREF(1)
( 1.0 0.0 -1.0 )
( 0.0 1.0 2.0 )
```

V menu sa vytvára kópia matice pred elimináciou.

#### 7. Inverzná matica

Výpočet inverznej matice.

**Vstup:**
```
Vyber operáciu: 7

Dostupné matice: 1, 2, 3, ...
Matica: 2
```

**Výstup (úspech):**
```
9 = 2^(-1)
( 1.0 0.0 0.0 )
( 0.0 1.0 0.0 )
( 0.0 0.0 1.0 )
```

**Výstup (chyba):**
```
Chyba: Matrix is not invertible (not regular)
```

Matica musí byť štvorcová a regulárna (s plnou hodnosťou).

#### 8. Riešiť sústavu rovníc

Riešenie sústavy lineárnych rovníc zadanej rozšírenou maticou `(A|b)`.

**Vstup:**

Sústava:
```
x + 2y = 5
3x + 4y = 11
```

Rozšírená matica:
```
Vyber operáciu: 8

Zadaj rozšírenú maticu sústavy (A|b):

Zadaj počet riadkov a stĺpcov (m n):
2 3
Zadaj 2 riadkov po 3 číslach:
1 2 5
3 4 11
```

**Výstup:**
```
Riešenie:
Parametrized solution set: (1.0, 2.0)
```

Znamená: `x = 1`, `y = 2`.

**Príklad s nekonečne veľa riešení:**

Sústava:
```
x + y = 3
2x + 2y = 6
```

```
Vyber operáciu: 8

Zadaj rozšírenú maticu sústavy (A|b):

Zadaj počet riadkov a stĺpcov (m n):
2 3
Zadaj 2 riadkov po 3 číslach:
1 1 3
2 2 6
```

**Výstup:**
```
Riešenie:
Parametrized solution set: (3.0, 0.0) + t0 * (-1.0, 1.0)
```

Znamená: `x = 3 - t₀`, `y = t₀`.

**Príklad bez riešenia:**
```
Riešenie:
No solution.
```

#### 9. Info o matici

Zobrazí všetky informácie o matici.

**Vstup:**
```
Vyber operáciu: 9

Dostupné matice: 1, 2, 3, ...
Matica: 1
```

**Výstup:**
```
1:
( 1.0 2.0 3.0 )
( 4.0 5.0 6.0 )

Rozmery: 2 x 3
Hodnosť: 2
Štvorcová: False
```

Ak je matica štvorcová, zobrazí sa aj:
```
Regulárna: True
Symetrická: False
```

#### 0. Koniec

Ukončí program.

---

## 3. Reprezentácia výstupu

### Výpis matice

Matice sa vypisujú po riadkoch v zátvorkách:

```
( 1.0 2.0 3.0 )
( 4.0 5.0 6.0 )
```

Hodnoty sú zaokrúhlené na 10 desatinných miest pri výpise, ale vnútorne sa nezaokrúhľujú (môže nastať problém kvôli floating-point aritmetike).

### Riešenie sústavy

`get_solutions()` vracia text:

- **"No solution."** – sústava nemá riešenie
- **"Solution is the entire domain."** – degenerovaný prípad
- **"Parametrized solution set: (a, b, ...) + t0 * (c, d, ...) + t1 * (e, f, ...) + ..."** – parametrický zápis

---

## 4. Algoritmy a dátové štruktúry

**Dátová štruktúra matice:**
Hlavnou dátovou štruktúrou je trieda `Matrix`. 
Samotné dáta matice sú uložené ako 2D zoznam (zoznam zoznamov), 
kde každý vnútorný zoznam reprezentuje jeden riadok matice. 

**Maticový súčin:** Použitý bol štandardný algoritmus s časovou zložitosťou O(n³).



### Gaussova eliminácia – `ref()`

Gaussova eliminácia podľa skrípt pána prof. Fialy, mierne upravená pre praktické použitie:

1. Odstránenie nulových riadkov
2. Nájdenie pivotu pre každý riadok (prvý nenulový prvok v riadku)
3. Normalizácia každého riadku (tak, aby bol každý pivot rovný 1)
4. Eliminácia všetkých prvkov pod každým pivotom
5. Usporiadanie všetkých riadkov podľa pozície pivotu (počtu počiatočných núl)

Výstup: matica v REF

### Gauss-Jordanova eliminácia – `rref()`

Rozšírenie Gaussovej eliminácie:

1. Najprv zavolá `ref()`
2. Potom eliminuje aj prvky **nad** pivotmi

Výstup: matica v RREF

### Inverzia – `invert()`

Používa rozšírenú maticu `(A|I)`:

1. Vytvorí `(A|I)`, kde `I` je jednotková matica rozmeru `n`, ak A je rozmeru `n×n`
2. Aplikuje RREF na celú rozšírenú maticu
3. Ak je `A` regulárna: ľavá časť → `I`, pravá časť → `A⁻¹`
4. Inak: `ValueError`

### Riešenie sústavy – `get_solutions()`

Vstup: rozšírená matica `(A|b)`

1. RREF na `(A|b)`
2. Kontrola riešiteľnosti (napríklad riadok `0 0 ... 0 | b`, kde `b ≠ 0`)
3. Identifikácia voľných a bázických premenných
4. Konštrukcia afinného priestoru, prípadne iba jadra (ak homogénna sústava)
5. Výstup: parametrický popis riešení

---

## 5. Štruktúra programu

Podrobnejšiu dokumentáciu následne obsahujú popisy funkcií v samotnom programe. Tu je iba základný prehľad.

### Súbory

**`main.py`** (hlavný program)
- Hlavná (a jediná) trieda `Matrix`
- Menu handler:
  - `print_menu()` – vypíše menu s možnosťami
  - `read_matrix()` – načíta maticu od používateľa (rozmery a prvky)
  - `main()` – hlavná slučka programu, spracováva používateľské voľby a volá dané operácie

**`testovacia_sada.py`** (unit testy - testovacia sada)
- Pomocné funkcie:
  - `matrix_to_numpy(m: Matrix) -> np.ndarray` – konvertuje Matrix na numpy pole
  - `numpy_to_matrix(arr: np.ndarray) -> Matrix` – konvertuje numpy pole na Matrix
- Porovnanie výsledkov s numpy pre validáciu správnosti implementácie

### Hlavné metódy v `Matrix`

#### Vytváranie matice

- **`__init__(data: Iterable[Iterable[Number]])`** 
  - Hlavný konštruktor. Vytvára maticu z 2D poľa (zoznamu zoznamov). Vnútorne konvertuje na tuple pre nemennosť riadkov.
    - Príklad: `Matrix([[1, 2], [3, 4]])`

- **`zeros(m: int, n: int) -> Matrix`**
  - Vytvára nulovú maticu rozmerov m×n.
    - Príklad: `Matrix.zeros(3, 4)` vytvorí maticu 3×4 naplnenú nulami.

- **`identity(n: int) -> Matrix`**
  - Vytvára jednotkovú maticu n×n.
    - Príklad: `Matrix.identity(3)` vytvorí jednotkovú maticu 3×3.

- **`from_list(data: Iterable[Iterable[Number]]) -> Matrix`** 
  - Alias pre implicitný `__init__`, explicitné zadanie zoznamu.

- **`from_matrix(other: Matrix) -> Matrix`**
  - Vytvára hlbokú kópiu na základe inej matice.

- **`copy() -> Matrix`** 
  - Vytvorí novú hlbokú kópiu aktuálnej matice.

#### Vlastnosti - properties

- **`.rows: int`** – Počet riadkov matice.

- **`.cols: int`** – Počet stĺpcov matice.

- **`.shape: Tuple[int, int]`** – Rozmery matice `(rows, cols)`.

- **`.empty: bool`** – Či je matica prázdna (nemá žiadne riadky a žiadne stĺpce).

- **`.is_square: bool`** – Či je matica štvorcová (počet riadkov = počet stĺpcov).

- **`.is_regular: bool`** – Či je matica regulárna (je štvorcová a s plnou hodnosťou).

- **`.is_symmetric: bool`** – Či je matica symetrická (platí A = A^T).

#### Reprezentácia, indexovanie a iterácia

**Vždy, keď sa pristupuje k dátam, tak sa používa indexovanie od 1.**

- **`__str__() -> str`** – Vráti čitateľnú reprezentáciu matice v štandardnom tvare.
  - Každý riadok v zátvorkách, hodnoty zaokrúhlené na 10 desatinných miest.
  - Používa sa pri `print(matrix)`.

- **`__repr__() -> str`** – Vráti technickú reprezentáciu matice.
  - Formát: `Matrix(\n ... \n)`.

- **`__getitem__(key: Union[int, Tuple[int, int]]) -> Union[Tuple[Number, ...], Number]`** – Prístup k prvkom matice.
  - `A[i]` vracia i-ty riadok ako tuple (indexovanie **od 1**)
  - `A[i, j]` vracia prvok na pozícii (i, j) (indexovanie **od 1**)

- **`__setitem__(key: Union[int, Tuple[int, int]], value: Union[Iterable[Number], Number])`** – Nastavenie prvkov matice.
  - `A[i] = [1, 2, 3]` nastaví celý riadok
  - `A[i, j] = 5` nastaví konkrétny prvok

- **`__iter__() -> Iterator[Tuple[Number, ...]]`** – Iterovanie pomocou riadkov matice.
  - `for row in matrix: print(row)`

- **`__len__() -> int`** – Vracia počet riadkov matice (nie prvkov!).

#### Aritmetické operácie (vracajú nový objekt)

- **`__add__(other: Matrix) -> Matrix`** – Sčítanie matíc zľava `A + other`.
  - Matice musia mať rovnaký rozmer, inak `ValueError`.

- **`__radd__(other: Union[Number, Matrix]) -> Matrix`** – Sčítanie sprava `other + A`.
  - Podporuje aj `0 + A` (hodí sa napríklad pre `sum([A, B, C])`).

- **`__sub__(other: Matrix) -> Matrix`** – Odčítanie matíc sprava `A - other`.

- **`__rsub__(other: Union[Number, Matrix]) -> Matrix`** – Odčítanie zľava `other - A`.

- **`__mul__(other: Union[Number, Matrix]) -> Matrix`** – Skalárny násobok `A * other`.

- **`__rmul__(other: Number) -> Matrix`** – Násobenie sprava `other * A`.

- **`__matmul__(other: Matrix) -> Matrix`** – Maticový súčin `A @ other`.
  - Počet stĺpcov A musí byť rovný počtu riadkov B.

- **`__eq__(other: Union[object, Matrix]) -> bool`** – Porovnanie matíc `A == other`.
  - Porovnáva rozmery aj všetky prvky.

#### Transformácie (in-place)

- **`transpose() -> Matrix`** – Transponuje maticu na mieste (výmena riadkov a stĺpcov).

- **`ref() -> Matrix`** – Gaussova eliminácia. Prevedie maticu do REF (row echelon form).
  - Kroky: odstránenie nulových riadkov → normalizácia pivotov → eliminácia pod pivotmi → usporiadanie.

- **`rref() -> Matrix`** – Gauss-Jordanova eliminácia. Prevedie maticu do RREF (reduced row echelon form).
  - Najprv zavolá `ref()`, potom eliminuje aj nad pivotmi.

#### Výpočty a analýza

- **`rank() -> int`** – Hodnosť matice (počet pivotov v REF).
  - Vykonáva REF na kópii matice, takže je časovo netriviálne.

- **`invert() -> Matrix`** – Inverzia matice. Vracia novú maticu A^(-1).
  - Funguje len pre regulárne štvorcové matice.
  - Používa augmentovanú maticu (A|I), prevedie do (I | A^(-1)).
  - Vracia `ValueError` ak matica nie je štvorcová alebo regulárna.

- **`get_solutions() -> str`** – Rieši sústavu lineárnych rovníc zadanú ako rozšírená matica (A|b).
  - Vracia:
    - `"No solution."` – nemá riešenie
    - `"Solution is the entire domain."` – degenerovaný prípad
    - `"Parametrized solution set: ..."` – parametrický zápis riešení
  - Formát: `(a, b, c) + t0 * (d, e, f) + t1 * (g, h, i) + ...`

#### Elementárne riadkové operácie (ERO) (in-place)

- **`swap_rows(base: int, secondary: int) -> Matrix`** – Vymení pozície dvoch riadkov.
  - Indexovanie od 1.

- **`add_row(base: int, operand: int, multiple: Number = 1) -> Matrix`** 
  - Pripočíta násobok riadku `operand` ku riadku `base`.
  - `base[i] = base[i] + multiple * operand[i]` pre všetky stĺpce.

- **`subtract_row(base: int, operand: int, multiple: Number) -> Matrix`** 
  - Odpočíta násobok riadku `operand` od riadku `base`.
  - Je to ekvivalent `add_row(base, operand, -multiple)`.

- **`scale_row(base: int, multiple: Number) -> Matrix`** 
  - Vynásobí celý riadok nenulovým skalárom.

#### Práca s pivotmi a nulovými riadkami

- **`is_row_empty(row: int) -> bool`** – Testuje, či je daný riadok nulový (všetky prvky sú 0).

- **`remove_row(row: int) -> Matrix`** – Odstráni daný riadok z matice (in-place).
  - Posúva všetky nasledujúce riadky (riadky pod odstráneným riadkom) nahor (o 1) a zmenšuje rozmery matice.

- **`remove_null_rows() -> Matrix`** – Odstráni všetky nulové riadky z matice (in-place).

- **`get_pivot(row: int) -> Union[int, None]`** – Vráti index (stĺpec) prvého nenulového prvku v danom riadku (pivot).
  - Vracia `None` ak je riadok nulový alebo neexistuje.

- **`get_pivots(start_from_one: bool = True) -> List[int]`** – Vráti zoznam pozícií pivotov pre všetky riadky.
  - Parameter `start_from_one` určuje, či indexovanie začína od 1 (default) alebo 0.

- **`sort_by_pivots() -> Matrix`** – Usporiadá riadky matice podľa pozície pivotov vzostupne (podľa počtu počiatočných núl v riadku).
  - Nulové riadky sa posúvajú na koniec.

- **`eliminate_rows(base: int, below: bool = True) -> Matrix`** – Eliminuje prvky v stĺpci pivotu riadku `base`.
  - Ak `below=True`: eliminuje pod pivotom
  - Ak `below=False`: eliminuje nad pivotom

#### Validačné metódy (interné)

- **`_check_same_shape(other: Matrix) -> None`** 
  - Skontroluje, či majú dve matice rovnaký rozmer.
  - Vracia `ValueError` ak nie.

- **`_check_multiplication(other: Matrix) -> None`** 
  - Skontroluje, či je možné dve matice vynásobiť.
  - Overuje, či `self.cols == other.rows`.
  - Vracia `ValueError` ak nie.

---

## 6. Testovanie, unit testy

Spustenie testov:

```bash
python -m unittest testovacia_sada -v
```

alebo

```bash
python testovacia_sada.py
```

Testujú sa postupne (nie nutne v tomto poradí):
- Základné operácie (sčítanie, násobenie, transpozícia)
- Eliminácie (REF, RREF) – porovnanie s numpy
- Inverzia – overenie `A @ A⁻¹ = I`
- Riešenie sústavy – rôzne prípady (1 riešenie, ∞ riešení, 0 riešení)
- Okrajové prípady (prázdne matice, nekompatibilné rozmery, nulové riadky)

---

## 7. Možné rozšírenia do budúcna (ak bude čas a vôľa)

**Lepší input handling:**
- Validácia počtu zadaných čísel
- Lepšie error messages
- Import matíc zo súboru
- Lepšie demo, ktoré by umožňovalo viac funkcií

**Štruktúrovaný výstup:**
- aby `get_solutions()` vracalo výstup vhodný aj pre programy, nielen na čítanie
- Export do CSV/JSON

**Nová funkcionalita:**
- Determinant
- Vlastné čísla/vektory
- Počítanie nad konečnými telesami
- Počítanie permutácií

---

## 8. Ako sa na projekte pracovalo?

Nebudem klamať, projekt bol náročnejší ako som spočiatku predpokladal. Bolo potrebné definovať pomerne veľké množstvo nielen hlavných metód, ktoré sú to hlavné, čo nás zaujíma (ako napríklad G-J eliminácia), ale aj ďalšie množstvo rôznych helper funkcií, bez ktorých by bol kód veľmi neprehľadný a často by sa opakoval.
Na záver sa ukázalo, že písanie aj pomerne stručnej dokumentácie k jednotlivým metódam zaberie dosť času.

Vývoj prebiehal postupne: základné definície → základné operácie → Gaussova a Gauss-Jordanova eliminácia → riešenie sústav. \
Gaussova eliminácia bola zo všetkých algoritmov určite najzložitejšia časť celého programu, najmä kvôli voľbe správneho algoritmu a jeho nasledovnej implementácie. Bolo to o dosť náročnejšie ako pseudokód v našich skriptách, pretože bolo nutné definovať aj ďalšie pomocné metódy ako napríklad elementárne riadkové operácie.


---

## 9. Záver

Program síce nebol mojim prvým veľkým projektom v Pythone, ale určite to bol môj prvý projekt, kde som sa snažil použiť trochu viac "advanced" OOP princípy, typy (bol som na nich zvyknutý z TypeScriptu a mám rád IntelliSense) a tiež princípy TDD - Test Driven Developmentu, teda najprv som sa pokúšal si písať k funkciám unit-testy a až potom k nim písať implementáciu. Toto vyžadovalo aj to, aby som si na začiatku vývoja v podstate už rovno definoval všetky funkcie, ktoré som plánoval implementovať, čo určite pomohlo celkovej štruktúre kódu, ktorá je, podľa môjho názoru, celkom dobre zmáknutá.

Program nie je dokonalý (napríklad by to chcelo implementovať epsilon toleranciu pri floating point číslach), ale svoju funkciu plní a sám som ho pri opakovaní si LA1 často použil na overenie numerických výpočtov.
