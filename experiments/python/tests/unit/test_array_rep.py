from active_inference.array_rep import Rank, Dimensions, Domain, Field, PartalNormedField, Distro
from numpy import array
from numpy.testing import assert_allclose

TEST_FIELD_1 = Field.from_array(array([
    [[1, 2, 3, 4], [5, 1, 2, 2], [0, 4, 6, 0]],
    [[4, 3, 2, 1], [7, 1, 1, 1], [0, 0, 0, 1]],
]))

TEST_FIELD_NORM_2 = Field.from_array(array([
    [[.1, .2, .3, .4], [.5, .1, .2, .2], [0, .4, .6, 0]],
    [[.4, .3, .2, .1], [.7, .1, .1, .1], [0, 0, 0, 1]],
]))

TEST_FIELD_NORM_0_2 = Field.from_array(array([
    [[.05, .1, .15, .2], [.25, .05, .1, .1], [0, 4/11, 6/11, 0]],
    [[.2, .15, .1, .05], [.35, .05, .05, .05], [0, 0, 0, 1/11]],
]))

def test_field_domain() -> None:
    assert TEST_FIELD_1.domain == Domain.create(2, 3, 4)


def test_field_partially_normalize_single_dim() -> None:
    subspace = TEST_FIELD_1.domain.subspace(2)
    assert subspace is not None

    normed = PartalNormedField.normalize_subspace(TEST_FIELD_1, subspace)
    assert normed is not None

    assert_allclose(normed.field.value, TEST_FIELD_NORM_2.value)
    

def test_field_partially_normalize_multiple_dim() -> None:
    subspace = TEST_FIELD_1.domain.subspace(0, 2)
    assert subspace is not None
    
    normed = PartalNormedField.normalize_subspace(TEST_FIELD_1, subspace)
    assert normed is not None
    
    assert_allclose(normed.field.value, TEST_FIELD_NORM_0_2.value)
 

def test_field_normalize() -> None:
    assert_allclose(Distro.create(TEST_FIELD_1).normed_field.field.value, TEST_FIELD_1.value / 51)


