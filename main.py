from typing import Iterable, Iterator, Tuple, Union
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
            if not (1 <= key < self.rows):
                raise IndexError("Row index out of range")
            return tuple(self._data[key - 1])  # Apoužívame indexovanie od 1

        if isinstance(key, tuple):
            row, col = key
            if not (1 <= row <= self.rows) or not (1 <= col <= self.cols):
                raise IndexError("Index out of range")
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
                raise IndexError(f"Index [{self.rows}] of the row out of range of the matrix with shape {self.rows}*{self.cols}")
            if len(value_list) != self.cols:
                raise ValueError(f"Row length mismatch, given: {len(value_list)}, expected: {self.cols}")
            self._data[key - 1] = value_list # používame indexovanie od 1

        elif isinstance(key, tuple):
            row, col = key
            if not (1 <= row <= self.rows) or not (1 <= col <= self.cols):
                raise IndexError(f"Index [{self.rows}][{self.cols}] of the row out of range of the matrix with shape {self.rows}*{self.cols}")
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
        pass

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

        return self.is_square and self == self.transpose()

    # ===== Interné kontroly =====
    def _check_same_shape(self, other: "Matrix") -> None:
        pass

    def _check_multipliable(self, other: "Matrix") -> None:
        pass

    # ===== Operácie na matici =====
    def transpose(self) -> "Matrix":
        """
        Transponuje maticu a vráti novú. (Netransponuje na mieste, namiesto toho vytvorí novú maticu.)
        :return Nová transponovaná matica na základe tej aktuálnej.

        Napríklad:
            matica = Matrix([[1, 2, 3], [4, 5, 6]])
            print(matica.transpose())
            ( 1 4 )
            ( 2 5 )
            ( 3 6 )
        """
        tranposed_matrix = Matrix.zeros(self.cols, self.rows)
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                tranposed_matrix[col, row] = self[row, col]

        return tranposed_matrix

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
    def add_row(self, source: int, destination: int, multiple: Number) -> None:
        pass

    def subtract_row(self, source: int, destination: int, multiple: Number) -> None:
        return self.add_row(source, destination, -1 * multiple)

    def scale_row(self, source: int, multiple: Number) -> None:
        pass

    def swap_rows(self, source: int, destination: int) -> None:
        pass


class InputHandler:
    def __init__(self) -> None:
        pass

symetricka_matica = Matrix([[1, 2, 1], [1, 2, 1], [1, 2, 1]])
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

