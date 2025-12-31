"""
Jednoduchá maticová kalkulačka na lineárnu algebru.
Zápočtový program, zimný semester 2025/2026, Programovanie 1.
Peter Zaťko
"""

from numbers import Number
from typing import Iterable, Iterator, Tuple, Union, List


class Matrix:
    # ===== Inicializácia =====
    def __init__(self, data: Iterable[Iterable[Number]]):
        """
        Inicializuje maticu prostredníctvom vstupného 2D listu. Je to ekvivalentné volaniu metódy from_list.
        :param data: 2D list
        """
        # vytvárame úplnú kópiu dát, aby sme nemali len referenciu
        self._data = [list(row) for row in data]

    @classmethod
    def zeros(cls, m: int, n: int) -> "Matrix":
        """
        Inicializuje maticu tak, že z nej vytvorí maticu samých núl s rozmerni m riadkov * n stĺpcov.
        :param m: počet riadkov
        :param n: počet stĺpcov
        :return:
        """
        # celá matica je zaplnená samými nulami
        data = [[0 for col in range(n)] for row in range(m)]
        return cls(data)

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        """
        Inicializuje maticu tak, že z nej vytvorí maticu identity s rozmermi n riadkov * n stĺpcov.
        :param n: počet riadkov a počet stĺpcov
        :return: Zinicializovaná matica
        """
        # všade 0
        # okrem prípadu, kedy row = col, vtedy 1 (podľa definície jednotkovej matice)
        data = [[(1 if row == col else 0) for col in range(n)] for row in range(n)]
        return cls(data)

    @classmethod
    def from_list(cls, data: Iterable[Iterable[Number]]) -> "Matrix":
        """
        Inicializuje maticu prostredníctvom vstupného 2D listu. Je to ekvivalentné volaniu priamo constructora.
        :param data: 2D list
        :return: Zinicializovaná matica
        """
        return cls(data)

    @classmethod
    def from_matrix(cls, other: "Matrix") -> "Matrix":
        """
        Vytvorí novú maticu ako kópiu existujúcej.
        :param other: existujúca matica
        :return: nová, nezávislá matica
        """
        new_data = [row[:] for row in other._data]  # kópia dát matice
        return cls(new_data)

    # ===== Reprezentácia =====
    def __str__(self) -> str:
        """
        Vráti obsah matice v čítateľnom formáte.
        :return: str
        """
        out = ""
        first = True
        # vypíše obsah matice v štandardnom čítateľnom formáte
        # každý riadok v zátvorkách, hodnoty oddelené medzerou
        for row in self:
            if first:
                first = False
            else:
                out += "\n"
            out += "("
            for column in row:
                out += f" {round(column, 10)}"
            out += " )"

        return out if out != "" else "()"

    def __repr__(self) -> str:
        return f"Matrix(\n{self.__str__()}\n)"

    # ===== Dáta =====
    @property
    def shape(self) -> Tuple[int, int]:
        """
        :return: Tvar matice ako tuple (m, n)
        """
        return self.rows, self.cols

    @property
    def rows(self) -> int:
        """
        :return: Počet riadkov v matici, ak tam nejaké sú, inak 0.
        """
        return len(self)

    @property
    def cols(self) -> int:
        """
        :return: Počet stĺpcov v matici, ak tam nejaké sú, inak 0.
        """
        if self.rows < 1:
            return 0

        return len(self._data[0])

    @property
    def empty(self) -> bool:
        """
        Funkcia vyhodnotí, či je matica prázdna (0 riadkov).
        :return: Pravdivostná hodnota typu bool.
        """
        if self.rows < 1:
            return True

        return False

    def __len__(self) -> int:
        """
        :return: Vráti počet riadkov v matici.
        """
        if not self._data:
            return 0

        return len(self._data)

    def __getitem__(self, key: Union[int, tuple[int, int]]) -> Union[tuple[Number, ...], Number, None]:
        """
        :param key: Kĺúč v tvare [riadok] alebo [riadok][stĺpec]
        :return: Tuple (vektor) na riadku [riadok], prípadne číselný prvok na riadku [riadok] v stĺpci [stĺpec].
        """
        if self.empty:
            return None

        if isinstance(key, int):
            if not (1 <= key <= self.rows):
                raise IndexError(f"Row index {key} out of range")
            return tuple(self._data[key - 1])  # Apoužívame indexovanie od 1

        if isinstance(key, tuple):
            row, col = key
            if not (1 <= row <= self.rows) or not (1 <= col <= self.cols):
                raise IndexError(f"Index [{row}][{col}] out of range")
            return self._data[row - 1][col - 1]  # používame indexovanie od 1

        raise TypeError("Invalid key type")

    def __setitem__(self, key: Union[int, tuple[int, int]], value: Union[Iterable[Number], Number]) -> None:
        """
        :param key: Kĺúč v tvare [riadok] alebo [riadok][stĺpec]
        :param value: Tuple (vektor), jednotlivý číselný prvok.
        :return: None.
        """
        if isinstance(key, int):
            value_list = list(value)  # explicitná kópia
            if not (1 <= key <= self.rows):
                raise IndexError(
                    f"Index [{key}] out of the matrix with shape {self.rows}*{self.cols}")
            if len(value_list) != self.cols:
                raise ValueError(f"Row length mismatch, given: {len(value_list)}, expected: {self.cols}")
            self._data[key - 1] = value_list  # používame indexovanie od 1

        elif isinstance(key, tuple):
            row, col = key
            if not (1 <= row <= self.rows) or not (1 <= col <= self.cols):
                raise IndexError(
                    f"Index [{row}][{col}] out of the matrix with shape {self.rows}*{self.cols}")
            self._data[row - 1][col - 1] = value  # používame indexovanie od 1

        else:
            raise TypeError(f"Invalid key type, given: {type(key)}, expected: tuple[int, int]")

    def __iter__(self) -> Iterator[tuple[Number, ...]]:
        """
        Umožňuje iterovať maticu intuitívnym spôsobom, napríklad:
        matica = Matrix()
        for riadok in matica
             for stlpec in riadok:
                 print(stlpec)
        """
        for row in self._data:
            yield tuple(row)  # immutable row

    def copy(self) -> "Matrix":
        """
        Vytvorí novú maticu ako kópiu tej aktuálnej.
        :return: Nová matica, ktorá je kópiou pôvodnej matice.
        """
        new_matrix = Matrix.from_matrix(self)
        return new_matrix

    # ===== Charakterizácia matice =====
    def rank(self) -> int:
        """
        Najprv prevedie maticu do REF a následne vypočíta jej hodnosť.
        Upozornenie: Táto operácia je časovo náročná, keďže musí vždy vykonať Gaussovu elimináciu!
        :return: Hodnosť matice.
        """
        if self.empty:
            return 0

        # vytvor kópiu matice
        m = self.copy()

        # preveď kópiu na REF
        m.ref()

        # počet riadkov REF = rank matice
        return m.rows

    @property
    def is_square(self) -> bool:
        """
        Zistí, či je matica štvorcová.
        :return: Pravdivostná hodnota typu bool.
        """
        return self.rows == self.cols

    @property
    def is_regular(self) -> bool:
        """
        Matica je regulárna práve vtedy, ak je štvorcová a jej hodnosť sa rovná jej rozmerom.
        :return: Pravdivostná hodnota typu bool.
        """
        return self.rank() == self.cols == self.rows

    @property
    def is_symmetric(self) -> bool:
        """
        Ak je matica štvorcová, tak ju transponuje a následne ju porovná s netransponovanou maticou.
        Ak sa zhodujú -> matica je symetrická.
        :return: Pravdivostná hodnota typu bool.
        """
        if not self.is_square:
            return False

        transposed_matrix = Matrix.from_matrix(self)
        transposed_matrix.transpose()
        return self.is_square and self == transposed_matrix

    # ===== Interné kontroly =====
    def _check_same_shape(self, other: "Matrix") -> None:
        """
        Skontroluje, či majú dve matice rovnaký tvar.
        :param other: Matica na porovnanie.
        :raises ValueError: Ak matice nemajú rovnaký tvar.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Matrices must have the same shape: {self.shape} vs {other.shape}")

    def _check_multiplication(self, other: "Matrix") -> None:
        """
        Skontroluje, či je možné dve matice vynásobiť.
        :param other: Matica na porovnanie.
        :raises ValueError: Ak matice nie je možné vynásobiť.
        """
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply matrices with shapes {self.shape} and {other.shape}")

    # ===== Operácie na matici =====
    def transpose(self) -> None:
        """
        Transponuje maticu na mieste.

        Napríklad:
            matica = Matrix([[1, 2, 3], [4, 5, 6]])
            matica.transpose()
            print(matica)
            ( 1 4 )
            ( 2 5 )
            ( 3 6 )
        """

        if self.is_square:
            for row in range(1, self.rows + 1):
                # row + 1, aby sme zabránili dvojitému prehadzovaniu
                for col in range(row + 1, self.cols + 1):
                    self[col, row], self[row, col] = self[row, col], self[col, row]
        else:
            # Potrebujeme zmeniť rozmery matice
            new_data = [[0 for col in range(self.rows)] for row in range(self.cols)]
            for row in range(1, self.rows + 1):
                for col in range(1, self.cols + 1):
                    new_data[col - 1][row - 1] = self[row, col]
            self._data = new_data

    def is_row_empty(self, row) -> bool:
        """
        Zistí, či je daný riadok matice nulový - prázdny.
        :param row: Riadok, ktorý chceme testovať.
        :return: Pravdivostná hodnota typu bool.
        """
        if self.empty or row < 1 or row > self.rows:
            return False

        empty = True
        for col in self[row]:
            if col != 0:
                empty = False
                break

        return empty

    def remove_row(self, row) -> "Matrix":
        """
        Odstráni riadok z matice.
        :param row: Index riadku na odstránenie.
        """
        # posuň všetky riadky zdola nahor
        for move_up_row in range(row + 1, self.rows + 1):
            self[move_up_row - 1] = tuple(self[move_up_row])

        # zmenši rozmery matice o prázdny riadok
        self._data = self._data[:self.rows - 1]

        return self

    def remove_null_rows(self) -> "Matrix":
        """
        Odstráni všetky nulové riadky z matice.
        """
        if self.empty:
            return self

        empty_rows = 0
        for row in range(1, self.rows + 1):
            is_empty = self.is_row_empty(row - empty_rows)
            # posuň všetky riadky zdola nahor
            if is_empty:
                self.remove_row(row - empty_rows)
                empty_rows += 1

        return self

    def get_pivot(self, row) -> Union[int, None]:
        """
        Získa pozíciu pivotu v danom riadku matice. 0 znamená, že pivot neexistuje.
        :param row: Riadok matice, v ktorom chceme nájsť pivot.
        :return: Pozícia pivotu (index stĺpca) v danom riadku alebo None, ak je matica prázdna alebo riadok neexistuje.
        """
        if self.empty or row > self.rows:
            return None

        for i in range(1, self.cols + 1):
            possible_pivot = self[row, i]
            if possible_pivot != 0:
                return i

    def get_pivots(self, start_from_one: bool = True) -> List[int]:
        """
        Získa pozície pivotov v jednotlivých riadkoch matice.
        :param start_from_one: Či má indexovanie pivotov začínať od 1 (default), alebo od 0.
        :return: Zoznam pozícií pivotov v jednotlivých riadkoch.
        """
        pivots = []
        for row in range(1, self.rows + 1):
            pivot = self.get_pivot(row)
            pivots.append(pivot if start_from_one else pivot - 1)

        return pivots

    def sort_by_pivots(self) -> "Matrix":
        """
        Zoradí riadky matice podľa pozície pivotov.
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        if self.empty:
            return self

        pivots = self.get_pivots(False)

        for i in range(self.rows):
            for j in range(i + 1, self.rows):

                pi = pivots[i]
                pj = pivots[j]

                # None považujeme za "nekonečno"
                if pi is None:
                    pi = float("inf")
                if pj is None:
                    pj = float("inf")

                if pj < pi:
                    # prehoď pivoty
                    pivots[i], pivots[j] = pivots[j], pivots[i]

                    # prehoď riadky
                    self.swap_rows(i + 1, j + 1)

        return self

    def eliminate_rows(self, base, below=True) -> "Matrix":
        """
        Eliminuje riadky v matici.
        :param base: Riadok, ktorý obsahuje pivot.
        :param below: Ak True, eliminuje pod pivotom, inak nad pivotom.
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        if base > self.rows:
            return self

        base_pivot = self.get_pivot(base)
        if not base_pivot:
            return self

        pivot = self[base, base_pivot]

        # Naškáluj riadok tak, aby bol pivot 1
        if pivot != 1:
            self.scale_row(base, 1 / pivot)

        if below:
            for secondary in range(base + 1, self.rows + 1):
                self.subtract_row(secondary, base, self[secondary, base_pivot] / self[base, base_pivot])
        else:
            for secondary in range(1, base):
                self.subtract_row(secondary, base, self[secondary, base_pivot] / self[base, base_pivot])

        return self

    def ref(self) -> "Matrix":
        """
        Prevedie maticu do riadkovo-odstupňovaného tvaru = row echelon form (REF).
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        if self.empty:
            return self

        # odstráň všetky nulové riadky
        self.remove_null_rows()

        # zoraď riadky podľa pozície pivotov
        self.sort_by_pivots()

        # eliminuj všetky nenulové pozície pod pivotom v aktuálnom riadku
        for row in range(1, self.rows + 1):
            self.eliminate_rows(row)

        # odstráň nulové riadky
        self.remove_null_rows()

        return self

    def rref(self) -> "Matrix":
        """
        Prevedie maticu do redukovaného riadkovo-odstupňovaného tvaru = reduced row echelon form (RREF).
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """

        # preveď maticu do REF
        self.ref()

        # eliminuj všetky nenulové pozície nad pivotom v aktuálnom riadku
        for row in range(self.rows, 0, -1):
            self.eliminate_rows(row, False)

        # odstráň nulové riadky
        self.remove_null_rows()

        return self

    def get_solutions(self) -> str:
        self.rref()

        if self.empty:
            return "Solution is the entire domain."

        pivot_cols = self.get_pivots()

        # Riešiteľnosť
        for row in range(1, self.rows + 1):
            pivot = self.get_pivot(row)
            if pivot == self.cols: return "No solution."

        # Voľné premenné
        num_vars = self.cols - 1
        all_vars = set(range(1, num_vars + 1))
        free_vars = sorted(list(all_vars - set(pivot_cols)))

        # Bijekcia pivot_col -> row_index (pivot -> riadok)
        pivot_to_row = {}
        for row_idx in range(1, self.rows + 1):
            pivot_col = self.get_pivot(row_idx)
            if pivot_col:
                pivot_to_row[pivot_col] = row_idx

        # Špecifické riešenie
        part_sol = [0.0] * num_vars
        for p_col in pivot_cols:
            row_idx = pivot_to_row[p_col]
            part_sol[p_col - 1] = self[row_idx, self.cols]

        # Báza jadra (báza ker)
        kernel_base = []
        for f_col in free_vars:
            vec = [0.0] * num_vars
            vec[f_col - 1] = 1.0
            for p_col in pivot_cols:
                row_idx = pivot_to_row[p_col]
                vec[p_col - 1] = -1.0 * self[row_idx, f_col]
            kernel_base.append(vec)

        # Výstup
        output = f"Parametrized solution set: ({', '.join(str(round(n, 4)) for n in part_sol)})"
        for i, base in enumerate(kernel_base):
            output += f" + t{i} * ({', '.join(str(round(n, 4)) for n in base)})"
        return output


    def invert(self) -> "Matrix":
        """
        Vypočíta inverznú maticu.
        :return: Nová matica, ktorá je inverzná k pôvodnej matici (nová matica).
        :raises ValueError: Ak matica nie je regulárna (štvorcová s plnou hodnosťou).
        """
        if not self.is_regular:
            raise ValueError("Matrix is not invertible (not regular)")

        # Vytvoríme rozšírenú maticu [A|I]
        n = self.rows
        augmented = Matrix.zeros(n, 2 * n)

        # Na ľavú stranu dáme maticu A
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                augmented[i, j] = self[i, j]

        # Na pravú stranu dáme jednotkovú maticu
        for i in range(1, n + 1):
            augmented[i, i + n] = 1

        # Vykonáme G-J elimináciu
        augmented.rref()

        # Na pravej strane rozšírenej matice je naša požadovaná inverzná matica
        inverse = Matrix.zeros(n, n)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                inverse[i, j] = augmented[i, j + n]

        return inverse

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """
        Vynásobí dve matice (maticový súčin).
        :param other: Druhý činiteľ (matica).
        :return: Nová matica, ktorá je výsledkom súčinu.
        """
        if self.empty or other.empty:
            return Matrix([])

        self._check_multiplication(other)

        result = Matrix.zeros(self.rows, other.cols)
        for i in range(1, self.rows + 1):
            for j in range(1, other.cols + 1):
                sum_val = 0
                for k in range(1, self.cols + 1):
                    sum_val += self[i, k] * other[k, j]
                result[i, j] = sum_val

        return result

    def __eq__(self, other: Union[object, "Matrix"]) -> bool:
        """
        Porovná dve matice, či sú rovnaké vo všetkých zložkách.
        :param other: Matica, ktorú chceme porovnať s tou aktuálnou.
        :return: Pravdivostná hodnota typu bool.
        """
        # Ak sa jedná o iný dátový typ
        if not isinstance(other, Matrix):
            # Ak je matica prázdna, tak to budeme brať ako ekvivalentné hodnote None
            if self.empty and not other:
                return True

            return False

        # Triviálne prípady
        if self.empty and other.empty:
            return True
        if self.rows != other.rows or self.cols != other.cols:
            return False

        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                if self[row, col] != other[row, col]:
                    return False

        return True

    # ===== Operácie zľava =====
    def __add__(self, other: "Matrix") -> "Matrix":
        """
        Sčíta dve matice.
        :param other: Druhá matica na sčítanie.
        :return: Nová matica, ktorá je výsledkom sčítania.
        """
        if self.empty and other.empty:
            return Matrix([])

        self._check_same_shape(other)

        result = Matrix.zeros(self.rows, self.cols)
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                result[row, col] = self[row, col] + other[row, col]

        return result

    def __sub__(self, other: "Matrix") -> "Matrix":
        """
        Odčíta druhú maticu od prvej.
        :param other: Druhá matica na odčítanie.
        :return: Nová matica, ktorá je výsledkom odčítania.
        """
        if self.empty and other.empty:
            return Matrix([])

        self._check_same_shape(other)

        result = Matrix.zeros(self.rows, self.cols)
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                result[row, col] = self[row, col] - other[row, col]

        return result

    def __mul__(self, other: Union[Number, "Matrix"]) -> "Matrix":
        """
        Vynásobí maticu skalárom (skalárny násobok matice).
        :param other: Skalár.
        :return: Nová matica, ktorá je výsledkom skalárneho násobku.
        """
        if isinstance(other, Number):
            if self.empty:
                return Matrix([])

            result = Matrix.zeros(self.rows, self.cols)
            for row in range(1, self.rows + 1):
                for col in range(1, self.cols + 1):
                    result[row, col] = self[row, col] * other

            return result
        else:
            # If other is a Matrix, use matrix multiplication
            return self.__matmul__(other)

    # ===== Operácie zprava =====
    def __radd__(self, other: Union[Number, "Matrix"]) -> "Matrix":
        """
        Sčíta skalár a maticu (skalár + matica).
        :param other: Skalár na sčítanie.
        :return: Nová matica, ktorá je výsledkom sčítania.
        """
        if isinstance(other, Number):
            if self.empty:
                return Matrix([])

            result = Matrix.zeros(self.rows, self.cols)
            for row in range(1, self.rows + 1):
                for col in range(1, self.cols + 1):
                    result[row, col] = other + self[row, col]

            return result
        else:
            # If other is a Matrix, use regular addition
            return self.__add__(other)

    def __rsub__(self, other: Union[Number, "Matrix"]) -> "Matrix":
        """
        Odčíta maticu od skaláru (skalár - matica).
        :param other: Skalár, od ktorého sa odčíta matica.
        :return: Nová matica, ktorá je výsledkom odčítania.
        """
        if isinstance(other, Number):
            if self.empty:
                return Matrix([])

            result = Matrix.zeros(self.rows, self.cols)
            for row in range(1, self.rows + 1):
                for col in range(1, self.cols + 1):
                    result[row, col] = other - self[row, col]

            return result
        else:
            # If other is a Matrix, use regular subtraction with reversed operands
            return other.__sub__(self)

    def __rmul__(self, other: Number) -> "Matrix":
        """
        Vynásobí maticu skalárom (skalárny násobok matice).
        :param other: Skalár.
        :return: Nová matica, ktorá je výsledkom skalárneho násobku.
        """
        # Scalar multiplication is commutative, so we can use __mul__
        return self.__mul__(other)

    # ===== Elementárne riadkové úpravy (in-place) =====
    def add_row(self, base: int, operand: int, multiple: Number = 1) -> "Matrix":
        """
        Pripočíta ku riadku destination riadok source.
        :param base: Riadok, ku ktorému sa má pripočítať iný riadok.
        :param operand: Riadok, ktorý budeme pripočívať ku inému riadku.
        :param multiple: Skalár vyjadrujúci násobok riadku source.
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        if self.empty or operand > self.rows or base > self.rows:
            return self

        for col in range(1, self.cols + 1):
            self[base, col] += multiple * self[operand, col]

        return self

    def subtract_row(self, base: int, operand: int, multiple: Number) -> "Matrix":
        """
        Odpočíta od riadku destination riadok source. Je to ekvivalentné add_row s multiple = -1.
        :param base: Riadok, od ktorého sa má odpočítať iný riadok.
        :param operand: Riadok, ktorý budeme odpočítavať od iného riadku.
        :param multiple: Skalár vyjadrujúci násobok riadku source.
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        return self.add_row(base, operand, -1 * multiple)

    def scale_row(self, base: int, multiple: Number) -> "Matrix":
        """
        Vynásobí riadok nenulovým číslom.
        :param base: Riadok, nad ktorým chceme vykonať operáciu.
        :param multiple: Skalár vyjadrujúci násobok riadku source.
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        if self.empty or base > self.rows or multiple == 0:
            return self

        for col in range(1, self.cols + 1):
            self[base, col] = multiple * self[base, col]

        return self

    def swap_rows(self, base: int, secondary: int) -> "Matrix":
        """
        Vymení pozície dvoch riadkov.
        :param secondary: Index 1. riadku pre výmenu.
        :param base: Index 2. riadku pre výmenu.
        :return: Referenciu na túto maticu (self) pre reťazenie metód.
        """
        if self.empty or secondary > self.rows or base > self.rows:
            return self

        for col in range(1, self.cols + 1):
            self[secondary, col], self[base, col] = self[base, col], self[secondary, col]

        return self


class InputHandler:
    def __init__(self) -> None:
        pass


# data = [[1, 2, 0, -1, 5],
#         [0, 0, 1, 3, 2],
#         [0, 0, 0, 0, 0]]
# m = Matrix(data)
# print(m.get_solutions())
