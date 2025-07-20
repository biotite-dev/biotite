import numpy as np
import pytest
import biotite.structure as struc
from biotite.structure import query
from biotite.structure.bonds import BondList


@pytest.fixture
def atom_array():
    """Create a test AtomArray with diverse annotations."""
    chain_id = ["A", "A", "B", "B", "B", "C", "C"]
    res_id = [1, 1, 2, 2, 3, 3, 3]
    ins_code = ["", "", "", "A", "", "", ""]
    res_name = ["ALA", "ALA", "GLY", "GLY", "PRO", "PRO", "PRO"]
    hetero = [False, False, False, False, False, True, False]
    atom_name = ["N", "CA", "N", "CA", "N", "CA", "C"]
    element = ["N", "C", "N", "C", "N", "C", "C"]

    atom_list = []
    for i in range(7):
        atom_list.append(
            struc.Atom(
                [i, i + 1, i + 2],
                chain_id=chain_id[i],
                res_id=res_id[i],
                ins_code=ins_code[i],
                res_name=res_name[i],
                hetero=hetero[i],
                atom_name=atom_name[i],
                element=element[i],
            )
        )
    array = struc.array(atom_list)

    # Add some NaN coordinates to test has_nan_coord
    array.coord[3] = [np.nan, np.nan, np.nan]

    # Add bonds to test has_bonds function
    bonds = BondList(array.array_length())
    bonds.add_bond(0, 1)  # N-CA bond in residue 1
    bonds.add_bond(2, 3)  # N-CA bond in residue 2
    bonds.add_bond(4, 5)  # N-CA bond in residue 3
    array.bonds = bonds

    return array


@pytest.fixture
def atom_array_stack(atom_array):
    """Create a test AtomArrayStack."""
    array2 = atom_array.copy()
    array2.coord += 10
    array3 = atom_array.copy()
    array3.coord += 20
    return struc.stack([atom_array, array2, array3])


class TestQueryExpression:
    """Test QueryExpression class functionality."""

    def test_initialization(self):
        """Test QueryExpression initialization."""
        expr = query.QueryExpression("chain_id == 'A'")
        assert expr.expr == "chain_id == 'A'"
        assert expr.tree is not None

    def test_simple_equality(self, atom_array):
        """Test simple equality queries."""
        expr = query.QueryExpression("chain_id == 'A'")
        result = expr.query(atom_array)
        expected_indices = [0, 1]  # First two atoms are in chain A
        assert result.array_length() == len(expected_indices)
        assert (result.chain_id == "A").all()

    def test_inequality(self, atom_array):
        """Test inequality queries."""
        expr = query.QueryExpression("res_id != 1")
        result = expr.query(atom_array)
        assert (result.res_id != 1).all()
        assert result.array_length() == 5  # 7 total - 2 from res_id=1

    def test_comparison_operators(self, atom_array):
        """Test comparison operators (<, <=, >, >=)."""
        # Test greater than
        expr = query.QueryExpression("res_id > 1")
        result = expr.query(atom_array)
        assert (result.res_id > 1).all()

        # Test less than or equal
        expr = query.QueryExpression("res_id <= 2")
        result = expr.query(atom_array)
        assert (result.res_id <= 2).all()

    def test_logical_operators(self, atom_array):
        """Test logical AND (&) and OR (|) operators."""
        # Test AND with &
        expr = query.QueryExpression("(chain_id == 'A') & (atom_name == 'CA')")
        result = expr.query(atom_array)
        assert result.array_length() == 1
        assert result.chain_id[0] == "A"
        assert result.atom_name[0] == "CA"

        # Test OR with |
        expr = query.QueryExpression("(chain_id == 'A') | (chain_id == 'C')")
        result = expr.query(atom_array)
        assert result.array_length() == 4  # 2 from A + 2 from C

        # Test NOT with ~
        expr = query.QueryExpression("~(chain_id == 'A')")
        result = expr.query(atom_array)
        assert result.array_length() == 5  # 7 total - 2 from A
        assert not (result.chain_id == "A").any()

    def test_in_operator(self, atom_array):
        """Test 'in' and 'not in' operators."""
        # Test 'in' with list
        expr = query.QueryExpression("chain_id in ['A', 'C']")
        result = expr.query(atom_array)
        assert result.array_length() == 4
        assert set(result.chain_id.tolist()) == {"A", "C"}

        # Test 'not in'
        expr = query.QueryExpression("res_name not in ['ALA', 'GLY']")
        result = expr.query(atom_array)
        assert (result.res_name == "PRO").all()

    def test_functions(self, atom_array):
        """Test built-in functions like has_nan_coord and has_bonds."""
        # Test has_nan_coord
        expr = query.QueryExpression("has_nan_coord()")
        result = expr.query(atom_array)
        assert result.array_length() == 1  # Only one atom has NaN coords

        expr = query.QueryExpression("~has_nan_coord()")
        result = expr.query(atom_array)
        assert result.array_length() == 6  # 7 total - 1 with NaN

        # Test has_bonds
        expr = query.QueryExpression("has_bonds()")
        result = expr.query(atom_array)
        # Atoms 0,1,2,3,4,5 are involved in bonds (but atom 3 has NaN coords)
        expected_bonded = 6
        assert result.array_length() == expected_bonded

    def test_complex_queries(self, atom_array):
        """Test complex combined queries."""
        expr = query.QueryExpression(
            "(chain_id == 'A') & (atom_name == 'CA') & ~has_nan_coord()"
        )
        result = expr.query(atom_array)
        assert result.array_length() == 1
        assert result.chain_id[0] == "A"
        assert result.atom_name[0] == "CA"

        # Complex query with functions and operators
        expr = query.QueryExpression(
            "has_bonds() & (res_name in ['ALA', 'GLY']) & ~has_nan_coord()"
        )
        result = expr.query(atom_array)
        # Should get atoms 0, 1, 2 (atom 3 has NaN coords)
        assert result.array_length() == 3

    def test_mask_method(self, atom_array):
        """Test the mask method returns correct boolean array."""
        expr = query.QueryExpression("chain_id == 'A'")
        mask = expr.mask(atom_array)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (atom_array.array_length(),)
        assert mask.sum() == 2  # Two atoms in chain A
        assert mask[0] and mask[1] and not mask[2]

    def test_idxs_method(self, atom_array):
        """Test the idxs method returns correct indices."""
        expr = query.QueryExpression("chain_id == 'A'")
        indices = expr.idxs(atom_array)
        assert isinstance(indices, np.ndarray)
        assert indices.dtype == np.int64
        assert indices.tolist() == [0, 1]

    def test_atom_array_stack(self, atom_array_stack):
        """Test queries work with AtomArrayStack."""
        expr = query.QueryExpression("chain_id == 'A'")
        result = expr.query(atom_array_stack)
        assert isinstance(result, struc.AtomArrayStack)
        assert result.stack_depth() == atom_array_stack.stack_depth()
        assert result.array_length() == 2  # Two atoms in chain A

    def test_error_handling(self, atom_array):
        """Test error handling for invalid queries."""
        # Test undefined name
        expr = query.QueryExpression("undefined_attr == 'test'")
        with pytest.raises(NameError, match="Name 'undefined_attr' is not defined"):
            expr.query(atom_array)

        # Test undefined function
        expr = query.QueryExpression("undefined_func()")
        with pytest.raises(NameError, match="Function 'undefined_func' is not defined"):
            expr.query(atom_array)

        # Test function with arguments (not supported)
        expr = query.QueryExpression("has_nan_coord(True)")
        with pytest.raises(ValueError, match="does not accept arguments"):
            expr.query(atom_array)

        # Test invalid 'in' operand
        expr = query.QueryExpression("chain_id in 42")
        with pytest.raises(TypeError, match="is not iterable"):
            expr.query(atom_array)


class TestStandaloneFunctions:
    """Test standalone query functions."""

    def test_query_function(self, atom_array):
        """Test the standalone query function."""
        result = query.query(atom_array, "chain_id == 'A'")
        assert isinstance(result, struc.AtomArray)
        assert result.array_length() == 2
        assert (result.chain_id == "A").all()

    def test_mask_function(self, atom_array):
        """Test the standalone mask function."""
        mask = query.mask(atom_array, "chain_id == 'A'")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.sum() == 2

    def test_idxs_function(self, atom_array):
        """Test the standalone idxs function."""
        indices = query.idxs(atom_array, "chain_id == 'A'")
        assert isinstance(indices, np.ndarray)
        assert indices.tolist() == [0, 1]


class TestAtomArrayMethods:
    """Test query methods added to AtomArray and AtomArrayStack."""

    def test_atom_array_query_method(self, atom_array):
        """Test AtomArray.query method."""
        result = atom_array.query("chain_id == 'A'")
        assert isinstance(result, struc.AtomArray)
        assert result.array_length() == 2
        assert (result.chain_id == "A").all()

    def test_atom_array_mask_method(self, atom_array):
        """Test AtomArray.mask method."""
        mask = atom_array.mask("chain_id == 'A'")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.sum() == 2

    def test_atom_array_idxs_method(self, atom_array):
        """Test AtomArray.idxs method."""
        indices = atom_array.idxs("chain_id == 'A'")
        assert isinstance(indices, np.ndarray)
        assert indices.tolist() == [0, 1]

    def test_atom_array_stack_methods(self, atom_array_stack):
        """Test AtomArrayStack query methods."""
        result = atom_array_stack.query("chain_id == 'A'")
        assert isinstance(result, struc.AtomArrayStack)
        assert result.array_length() == 2

        mask = atom_array_stack.mask("chain_id == 'A'")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.sum() == 2

        indices = atom_array_stack.idxs("chain_id == 'A'")
        assert isinstance(indices, np.ndarray)
        assert indices.tolist() == [0, 1]


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_results(self, atom_array):
        """Test queries that return empty results."""
        expr = query.QueryExpression("chain_id == 'Z'")  # Non-existent chain
        result = expr.query(atom_array)
        assert result.array_length() == 0

        mask = expr.mask(atom_array)
        assert not mask.any()

        indices = expr.idxs(atom_array)
        assert len(indices) == 0

    def test_all_atoms_match(self, atom_array):
        """Test queries where all atoms match."""
        expr = query.QueryExpression("res_id >= 1")
        result = expr.query(atom_array)
        assert result.array_length() == atom_array.array_length()

        mask = expr.mask(atom_array)
        assert mask.all()

    def test_no_bonds(self):
        """Test has_bonds function on array without bonds."""
        atom = struc.Atom([0, 0, 0], chain_id="A")
        array = struc.array([atom])

        expr = query.QueryExpression("has_bonds()")
        result = expr.query(array)
        assert result.array_length() == 0

        expr = query.QueryExpression("~has_bonds()")
        result = expr.query(array)
        assert result.array_length() == 1

    def test_chained_comparisons(self, atom_array):
        """Test chained comparison operators."""
        expr = query.QueryExpression("1 <= res_id <= 2")
        result = expr.query(atom_array)
        expected_count = sum(1 <= rid <= 2 for rid in atom_array.res_id)
        assert result.array_length() == expected_count

    def test_scalar_to_array_broadcasting(self, atom_array):
        """Test scalar boolean results getting broadcast to array."""
        # This should create a scalar True result that gets broadcast
        expr = query.QueryExpression("True")
        mask = expr.mask(atom_array)
        assert mask.all()
        assert len(mask) == atom_array.array_length()

        expr = query.QueryExpression("False")
        mask = expr.mask(atom_array)
        assert not mask.any()
        assert len(mask) == atom_array.array_length()


class TestBuiltinFunctions:
    """Test built-in query functions in detail."""

    def test_has_nan_coord_edge_cases(self):
        """Test has_nan_coord with various coordinate patterns."""
        # Array with partial NaN
        array = struc.AtomArray(3)
        array.coord = np.array(
            [
                [1.0, 2.0, 3.0],  # No NaN
                [np.nan, 2.0, 3.0],  # Partial NaN
                [np.nan, np.nan, np.nan],  # All NaN
            ]
        )

        expr = query.QueryExpression("has_nan_coord()")
        result = expr.query(array)
        assert result.array_length() == 2  # Two atoms have NaN coords

        # Test with AtomArrayStack
        stack = struc.stack([array, array])
        result = expr.query(stack)
        assert result.array_length() == 2

    def test_has_bonds_edge_cases(self):
        """Test has_bonds with various bond configurations."""
        array = struc.AtomArray(4)
        array.coord = np.random.rand(4, 3)

        # Add bonds only to some atoms
        bonds = BondList(4)
        bonds.add_bond(0, 1)  # Atoms 0 and 1 are bonded
        # Atoms 2 and 3 have no bonds
        array.bonds = bonds

        expr = query.QueryExpression("has_bonds()")
        result = expr.query(array)
        assert result.array_length() == 2  # Only atoms 0 and 1

        expr = query.QueryExpression("~has_bonds()")
        result = expr.query(array)
        assert result.array_length() == 2  # Only atoms 2 and 3


@pytest.mark.parametrize(
    "query_str,expected_count",
    [
        ("chain_id == 'A'", 2),
        ("res_name == 'PRO'", 3),
        ("hetero == True", 1),
        ("atom_name == 'CA'", 3),
        ("element == 'N'", 3),
        ("ins_code == 'A'", 1),
        ("res_id == 3", 3),
    ],
)
def test_parametrized_queries(atom_array, query_str, expected_count):
    """Test various queries with parametrized inputs."""
    expr = query.QueryExpression(query_str)
    result = expr.query(atom_array)
    assert result.array_length() == expected_count


class TestStringRepresentation:
    """Test string representations of QueryExpression."""

    def test_str_and_repr(self):
        """Test __str__ and __repr__ methods."""
        expr_str = "chain_id == 'A'"
        expr = query.QueryExpression(expr_str)

        assert str(expr) == expr_str
        assert repr(expr) == f"QueryExpression('{expr_str}')"


class TestCompatibilityWithExistingCode:
    """Test that query functionality works well with existing Biotite patterns."""

    def test_with_filtering_and_slicing(self, atom_array):
        """Test query combined with traditional filtering."""
        # First filter with query, then with traditional indexing
        ca_atoms = atom_array.query("atom_name == 'CA'")
        chain_a_ca = ca_atoms[ca_atoms.chain_id == "A"]
        assert chain_a_ca.array_length() == 1

        # Combine with slicing
        first_two = atom_array[:2]
        result = first_two.query("chain_id == 'A'")
        assert result.array_length() == 2

    def test_with_structure_operations(self, atom_array):
        """Test query with structure operations like concatenation."""
        chain_a = atom_array.query("chain_id == 'A'")
        chain_b = atom_array.query("chain_id == 'B'")

        combined = chain_a + chain_b
        assert combined.array_length() == 5

        # Query the combined structure
        result = combined.query("atom_name == 'CA'")
        assert result.array_length() == 2

    def test_compound_queries(self, atom_array):
        """Test compound queries."""
        result = atom_array.query(
            "(chain_id == 'A') & (atom_name == 'CA') & ~has_nan_coord()"
        )
        assert result.array_length() == 1
        assert result.chain_id[0] == "A"
        assert result.atom_name[0] == "CA"

        idxs = atom_array.idxs(
            "(chain_id == 'A') & (atom_name == 'CA') & ~has_nan_coord()"
        )
        assert idxs.tolist() == [1]

    def test_compound_queries2(self, atom_array):
        atom_array.set_annotation("b_factor", np.array([10, 51, 93, 40, 50, 60, 70], dtype=np.float32))
        result = atom_array.query("(res_name in ['ALA', 'GLY']) & (b_factor > 50)")
        assert result.array_length() == 2
        assert result.res_name[0] == "ALA"
        assert result.res_name[1] == "GLY"
        assert result.b_factor[0] > 50
        assert result.b_factor[1] > 50