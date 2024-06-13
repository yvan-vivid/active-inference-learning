from dataclasses import dataclass
from typing import FrozenSet, Optional, Self, Tuple, List

from numpy import array, float64, ndarray
from numpy.typing import NDArray

@dataclass(frozen=True)
class Rank:
    value: int

    def __contains__(self, value: int) -> bool:
        return 0 <= value < self.value

    @classmethod
    def create(cls, value: int) -> Optional[Self]:
        return cls(value) if value >= 0 else None


@dataclass(frozen=True)
class Dimensions:
    rank: Rank
    dims: FrozenSet[int]

    def is_full(self) -> bool:
        return len(self.dims) == self.rank.value

    def like(self, *n: int) -> Optional[Self]:
        return self.from_dims(self.rank, *n)

    @classmethod
    def from_rank(cls, rank: Rank) -> Self:
        return cls(rank, frozenset(range(rank.value)))

    @classmethod
    def empty(cls, rank: Rank) -> Self:
        return cls(rank, frozenset())
    
    @classmethod
    def from_dims(cls, rank: Rank, *n: int) -> Optional[Self]:
        if all(k in rank for k in n):
            return cls(rank, frozenset(n))
        return None


@dataclass(frozen=True)
class Domain:
    shape: Tuple[int, ...]
    dims: Dimensions

    @property
    def rank(self) -> Rank:
        return self.dims.rank

    def subspace(self, *n: int) -> Optional[Dimensions]:
        return self.dims.like(*n)

    @classmethod
    def create(cls, *n: int) -> Self:
        return cls(tuple(n), Dimensions.from_rank(Rank(len(n))))


@dataclass(frozen=True)
class Field:
    value: NDArray[float64]
    domain: Domain

    @classmethod
    def from_array(cls, a: NDArray[float64]) -> Self:
        return cls(a, Domain.create(*a.shape))


@dataclass(frozen=True)
class PartalNormedField:
    field: Field
    normalized: Dimensions
    
    def is_distro(self) -> bool:
        return self.normalized.is_full()

    def expected(self, field: Field) -> Optional[Field]:
        pass
    
    @classmethod
    def normalize_full(cls, field: Field) -> Self:
        norm = field.value.sum(axis=tuple(field.domain.dims.dims), keepdims=True)
        return cls(Field.from_array(field.value / norm), field.domain.dims)

    @classmethod
    def normalize_subspace(cls, field: Field, subspace: Dimensions) -> Optional[Self]:
        s = field.domain.dims if subspace is None else subspace
        if s.rank != field.domain.rank:
            return None

        norm = field.value.sum(axis=tuple(s.dims), keepdims=True)
        return cls(Field.from_array(field.value / norm), s)


@dataclass(frozen=True)
class Distro:
    normed_field: PartalNormedField

    @classmethod
    def create(cls, field: Field) -> Self:
        return cls(PartalNormedField.normalize_full(field))


@dataclass(frozen=True)
class Cross:
    a: Domain
    b: Domain
    contractions: List[Tuple[int, int]]

    @classmethod
    def create(cls, a: Domain, b: Domain, *contractions: Tuple[int, int]) -> Optional[Self]:
        for ca, cb in contractions:
            if not (ca in a.rank and cb in b.rank and a.shape[ca] == b.shape[cb]):
                return None
        return cls(a, b, list(contractions))


def product(a: Field, b: Field, cross: Cross) -> Optional[Field]:
    return None


