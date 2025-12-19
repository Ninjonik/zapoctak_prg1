"""
Jednoduchá kalkulačka na lineárnu algebru.
Zápočtový program, zimný semester 2025/2026, Programovanie 1.
Peter Zaťko
"""

from typing import Iterable, Iterator, Tuple, Union, List
from numbers import Number


class Matrix:
    def __init__(self, data: Iterable[Iterable[Number]]):
        """
        Inicializuje maticu prostredníctvom vstupného 2D listu. Je to ekvivalentné volaniu metódy from_list.
        :param data: 2D list
        """
        # vytvárame kópiu, aby sme nemali len referenciu
        self._data = [list(row) for row in data]

    # ===== Konštrukcia =====
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
        # všade 0 okrem prípadu kedy row = col (podľa definície jednotkovej matice)
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
        # deep copy of the data
        new_data = [row[:] for row in other._data]  # shallow copy
        return cls(new_data)

    # ===== Reprezentácia =====
    def __str__(self) -> str:
        """
        Vráti obsah matice v čítateľnom formáte.
        :return: str
        """
        out = ""
        first = True
        for row in self:
            if first:
                first = False
            else:
                out += "\n"
            out += "("
            for column in row:
                out += f" {column}"
            out += " )"

        return out

    def __repr__(self) -> str:
        return "Matrix(...)"

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
            value_list = list(value) # explicitná kópia
            if not (1 <= key <= self.rows):
                raise IndexError(f"Index [{key}] of the row out of range of the matrix with shape {self.rows}*{self.cols}")
            if len(value_list) != self.cols:
                raise ValueError(f"Row length mismatch, given: {len(value_list)}, expected: {self.cols}")
            self._data[key - 1] = value_list # používame indexovanie od 1

        elif isinstance(key, tuple):
            row, col = key
            if not (1 <= row <= self.rows) or not (1 <= col <= self.cols):
                raise IndexError(f"Index [{row}][{col}] of the row out of range of the matrix with shape {self.rows}*{self.cols}")
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
        new_matrix = Matrix.from_matrix(self)
        return new_matrix

    # ===== Charakterizácia matice =====
    def rank(self) -> int:
        """
        Najprv prevedie maticu do REF a následne vypočíta jej hodnosť.
        Upozornenie: Táto operácia je časovo náročná, keďže musí vždy vykonať Gaussovu elimináciu!
        :return: Hodnosť matice.
        """
        pass

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
        pass

    def _check_multipliable(self, other: "Matrix") -> None:
        pass

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

        if self.is_regular:
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
        if self.empty or row < 1 or row > self.rows:
            return False

        empty = True
        for col in self[row]:
            if col != 0:
                empty = False
                break

        return empty

    def remove_row(self, row) -> None:
        # posuň všetky riadky zdola nahor
        for move_up_row in range(row + 1, self.rows + 1):
            self[move_up_row - 1] = tuple(self[move_up_row])

        # zmenši rozmery matice o prázdny riadok
        self._data = self._data[:self.rows - 1]

    def remove_null_rows(self) -> None:
        if self.empty:
            return None

        empty_rows = 0
        for row in range(1, self.rows + 1):
            is_empty = self.is_row_empty(row)
            # posuň všetky riadky zdola nahor
            if is_empty:
                empty_rows += 1
                self.remove_row(row)

    def get_pivot(self, row) -> Union[Number, None]:
        if self.empty or row > self.rows:
            return None

        for i in range(1, self.cols + 1):
            possible_pivot = self[row, i]
            if possible_pivot != 0:
                return i

    def get_pivots(self) -> List[int]:
        pivots = []
        for row in range(1, self.rows + 1):
            pivot = self.get_pivot(row)
            pivots.append(pivot - 1)

        return pivots

    def sort_by_pivots(self) -> None:
        pivots = self.get_pivots()
        # zoraď riadky podľa počtu pivotov
        for row1 in range(1, self.rows + 1):
            pivot1 = pivots[row1 - 1]

            for row2 in range(1, self.rows + 1):
                pivot2 = pivots[row2 - 1]

                if pivot2 < pivot1:
                    self.swap_rows(row1, row2)

    def eliminate_pivots(self, base, below = True) -> None:
        if base > self.rows:
            return None

        base_pivot = self.get_pivot(base)
        if not base_pivot:
            return None

        pivot = self[base, base_pivot]

        # Naškáluj riadok tak, aby bol pivot 1
        if pivot != 1:
            self.scale_row(base, 1 / pivot)

        if below:
            for secondary in range(base + 1, self.rows + 1):
                secondary_pivot = self.get_pivot(secondary)
                if secondary_pivot == base_pivot:
                    self.subtract_row(secondary, base, self[secondary, secondary_pivot] / self[base, base_pivot])
        else:
            for secondary in range(1, base):
                secondary_pivot = self.get_pivot(secondary)
                self.subtract_row(secondary, base, self[secondary, base_pivot] / self[base, base_pivot])

    def ref(self) -> None:
        """
        Prevedie maticu do riadkovo-odstupňovaného tvaru = row echelon form (REF).
        """
        if self.empty:
            return None

        # odstráň všetky nulové riadky
        self.remove_null_rows()

        # nájdi pivoty pre každý riadok
        pivots = self.get_pivots()

        # zoraď riadky podľa počtu pivotov
        self.sort_by_pivots()

        # eliminuj všetky nenulové pozície pod pivotom v aktuálnom riadku
        for row in range(1, self.rows + 1):
            self.eliminate_pivots(row)

    def rref(self) -> None:
        """
        Prevedie maticu do redukovaného riadkovo-odstupňovaného tvaru = reduced row echelon form (RREF).
        """
        # preveď maticu do REF
        self.ref()

        # eliminuj všetky nenulové pozície nad pivotom v aktuálnom riadku
        for row in range(self.rows, 0, -1):
            self.eliminate_pivots(row, False)


    def invert(self) -> "Matrix":
        pass

    def __matmul__(self, other: "Matrix") -> "Matrix":
        pass

    def __eq__(self, other: Union[object, "Matrix"]) -> bool:
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
        pass

    def __sub__(self, other: "Matrix") -> "Matrix":
        pass

    def __mul__(self, other: Union[Number, "Matrix"]) -> "Matrix":
        pass

    # ===== Operácie zprava =====
    def __radd__(self, other: "Matrix") -> "Matrix":
        pass

    def __rsub__(self, other: "Matrix") -> "Matrix":
        pass

    def __rmul__(self, other: Number) -> "Matrix":
        pass

    # ===== Elementárne riadkové úpravy (in-place) =====
    def add_row(self, base: int, operand: int, multiple: Number = 1) -> None:
        """
        Pripočíta ku riadku destination riadok source.
        :param base: Riadok, ku ktorému sa má pripočítať iný riadok.
        :param operand: Riadok, ktorý budeme pripočívať ku inému riadku.
        :param multiple: Skalár vyjadrujúci násobok riadku source.
        :return: None
        """
        if self.empty or operand > self.rows or base > self.rows:
            return None

        for col in range(1, self.cols + 1):
            self[base, col] += multiple * self[operand, col]

    def subtract_row(self, base: int, operand: int, multiple: Number) -> None:
        """
        Odpočíta od riadku destination riadok source. Je to ekvivalentné add_row s multiple = -1.
        :param base: Riadok, od ktorého sa má odpočítať iný riadok.
        :param operand: Riadok, ktorý budeme odpočítavať od iného riadku.
        :param multiple: Skalár vyjadrujúci násobok riadku source.
        :return: None
        """
        return self.add_row(base, operand, -1 * multiple)

    def scale_row(self, base: int, multiple: Number) -> None:
        """
        Vynásobí riadok nenulovým číslom.
        :param base: Riadok, nad ktorým chceme vykonať operáciu.
        :param multiple: Skalár vyjadrujúci násobok riadku source.
        :return: None
        """
        if self.empty or base > self.rows or multiple == 0:
            return None

        for col in range(1, self.cols + 1):
            self[base, col] = multiple * self[base, col]

    def swap_rows(self, base: int, secondary: int) -> None:
        """
        Vymení pozície dvoch riadkov.
        :param secondary: Index 1. riadku pre výmenu.
        :param base: Index 2. riadku pre výmenu.
        :return: None
        """
        if self.empty or secondary > self.rows or base > self.rows:
            return None

        for col in range(1, self.cols + 1):
            self[secondary, col], self[base, col] = self[base, col], self[secondary, col]


class InputHandler:
    def __init__(self) -> None:
        pass

"""
symetricka_matica = Matrix([[1, 1, -1], [1, 2, 0], [-1, 0, 5]])
matica = Matrix([[6, 3, 1], [3, 2, 9], [7, -4, 0]])
jednotkova_matica=Matrix.identity(4)
nulova_matica=Matrix.zeros(3, 5)

print(symetricka_matica)
print(matica)
print(jednotkova_matica)
print(nulova_matica)

print(symetricka_matica.shape)
print(matica.shape)
print(jednotkova_matica.shape)
print(nulova_matica.shape)

print(symetricka_matica.is_symmetric)
print(matica.is_symmetric)
print(jednotkova_matica.is_symmetric)
print(nulova_matica.is_symmetric)
"""
# Case 2: Matrix with a true zero row
m2 = Matrix([
    [1, 1, 1],
    [0, 0, 0],
    [0, 1, 2]
])
# This will likely raise: TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
m2.rref()
print("Case 2 (Zero Row):\n", m2)