use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PySet, PyTuple};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;

/// Rust reflection of the :class:`BondType` enum.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum BondType {
    Any = 0,
    Single = 1,
    Double = 2,
    Triple = 3,
    Quadruple = 4,
    AromaticSingle = 5,
    AromaticDouble = 6,
    AromaticTriple = 7,
    Coordination = 8,
    Aromatic = 9,
}

impl BondType {
    fn without_aromaticity(&self) -> Self {
        match self {
            BondType::AromaticSingle => BondType::Single,
            BondType::AromaticDouble => BondType::Double,
            BondType::AromaticTriple => BondType::Triple,
            BondType::Aromatic => BondType::Any,
            _ => *self,
        }
    }
}

#[pyfunction]
pub fn bond_type_members() -> HashMap<String, u8> {
    let mut map = HashMap::new();
    map.insert("ANY".to_string(), BondType::Any as u8);
    map.insert("SINGLE".to_string(), BondType::Single as u8);
    map.insert("DOUBLE".to_string(), BondType::Double as u8);
    map.insert("TRIPLE".to_string(), BondType::Triple as u8);
    map.insert("QUADRUPLE".to_string(), BondType::Quadruple as u8);
    map.insert(
        "AROMATIC_SINGLE".to_string(),
        BondType::AromaticSingle as u8,
    );
    map.insert(
        "AROMATIC_DOUBLE".to_string(),
        BondType::AromaticDouble as u8,
    );
    map.insert(
        "AROMATIC_TRIPLE".to_string(),
        BondType::AromaticTriple as u8,
    );
    map.insert("COORDINATION".to_string(), BondType::Coordination as u8);
    map.insert("AROMATIC".to_string(), BondType::Aromatic as u8);
    map
}

impl TryFrom<u8> for BondType {
    type Error = PyErr;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BondType::Any),
            1 => Ok(BondType::Single),
            2 => Ok(BondType::Double),
            3 => Ok(BondType::Triple),
            4 => Ok(BondType::Quadruple),
            5 => Ok(BondType::AromaticSingle),
            6 => Ok(BondType::AromaticDouble),
            7 => Ok(BondType::AromaticTriple),
            8 => Ok(BondType::Coordination),
            9 => Ok(BondType::Aromatic),
            _ => Err(exceptions::PyValueError::new_err(format!(
                "BondType {} is invalid",
                value
            ))),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for BondType {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let value: u8 = ob.extract()?;
        BondType::try_from(value)
    }
}

impl<'py> IntoPyObject<'py> for BondType {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let bond_type_class = py.import("biotite.structure")?.getattr("BondType")?;
        bond_type_class.call1((self as u8,))
    }
}

/// Internal representation of a single bond.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: BondType,
}

impl Bond {
    /// Create a new bond with normalized atom indices (smaller index first).
    /// Negative indices are resolved relative to `atom_count`.
    fn new(atom1: isize, atom2: isize, bond_type: BondType, atom_count: usize) -> PyResult<Self> {
        let a1 = to_positive_index(atom1, atom_count)?;
        let a2 = to_positive_index(atom2, atom_count)?;
        let (a1, a2) = if a1 <= a2 { (a1, a2) } else { (a2, a1) };
        Ok(Bond {
            atom1: a1,
            atom2: a2,
            bond_type,
        })
    }

    /// Check whether two bonds connect the same atoms, ignoring the bond type.
    fn equal_atoms(&self, other: &Bond) -> bool {
        self.atom1 == other.atom1 && self.atom2 == other.atom2
    }
}

/// __init__(atom_count, bonds=None)
///
/// A bond list stores indices of atoms
/// (usually of an :class:`AtomArray` or :class:`AtomArrayStack`)
/// that form chemical bonds together with the type (or order) of the
/// bond.
///
/// Internally the bonds are stored as *n x 3* :class:`ndarray`.
/// For each row, the first column specifies the index of the first
/// atom, the second column the index of the second atom involved in the
/// bond.
/// The third column stores an integer that is interpreted as member
/// of the the :class:`BondType` enum, that specifies the order of the
/// bond.
///
/// When indexing a :class:`BondList`, the index is not forwarded to the
/// internal :class:`ndarray`. Instead the indexing behavior is
/// consistent with indexing an :class:`AtomArray` or
/// :class:`AtomArrayStack`:
/// Bonds with at least one atom index that is not covered by the index
/// are removed, atom indices that occur after an uncovered atom index
/// move up.
/// Effectively, this means that after indexing an :class:`AtomArray`
/// and a :class:`BondList` with the same index, the atom indices in the
/// :class:`BondList` will still point to the same atoms in the
/// :class:`AtomArray`.
/// Indexing a :class:`BondList` with a single integer is equivalent
/// to calling :func:`get_bonds()`.
///
/// The same consistency applies to adding :class:`BondList` instances
/// via the '+' operator:
/// The atom indices of the second :class:`BondList` are increased by
/// the atom count of the first :class:`BondList` and then both
/// :class:`BondList` objects are merged.
///
/// Parameters
/// ----------
/// atom_count : int
///     A positive integer, that specifies the number of atoms the
///     :class:`BondList` refers to
///     (usually the length of an atom array (stack)).
///     Effectively, this value is the exclusive maximum for the indices
///     stored in the :class:`BondList`.
/// bonds : ndarray, shape=(n,2) or shape=(n,3), dtype=int, optional
///     This array contains the indices of atoms which are bonded:
///     For each row, the first column specifies the first atom,
///     the second row the second atom involved in a chemical bond.
///     If an *n x 3* array is provided, the additional column
///     specifies a :class:`BondType` instead of :attr:`BondType.ANY`.
///     By default, the created :class:`BondList` is empty.
///
/// Notes
/// -----
/// When initially providing the bonds as :class:`ndarray`, the input is
/// sanitized: Redundant bonds are removed, and each bond entry is
/// sorted so that the lower one of the two atom indices is in the first
/// column.
/// If a bond appears multiple times with different bond types, the
/// first bond takes precedence.
///
/// Examples
/// --------
///
/// Construct a :class:`BondList`, where a central atom (index 1) is
/// connected to three other atoms (index 0, 3 and 4):
///
/// >>> bond_list = BondList(5, np.array([(1,0),(1,3),(1,4)]))
/// >>> print(bond_list)
/// [[0 1 0]
///  [1 3 0]
///  [1 4 0]]
///
/// Remove the first atom (index 0) via indexing:
/// The bond containing index 0 is removed, since the corresponding atom
/// does not exist anymore. Since all other atoms move up in their
/// position, the indices in the bond list are decreased by one:
///
/// >>> bond_list = bond_list[1:]
/// >>> print(bond_list)
/// [[0 2 0]
///  [0 3 0]]
///
/// :class:`BondList` objects can be associated to an :class:`AtomArray`
/// or :class:`AtomArrayStack`.
/// The following snippet shows this for a benzene molecule:
///
/// >>> benzene = AtomArray(12)
/// >>> # Omit filling most required annotation categories for brevity
/// >>> benzene.atom_name = np.array(
/// ...     ["C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2", "H3", "H4", "H5", "H6"]
/// ... )
/// >>> benzene.bonds = BondList(
/// ...     benzene.array_length(),
/// ...     np.array([
/// ...         # Bonds between carbon atoms in the ring
/// ...         (0,  1, BondType.AROMATIC_SINGLE),
/// ...         (1,  2, BondType.AROMATIC_DOUBLE),
/// ...         (2,  3, BondType.AROMATIC_SINGLE),
/// ...         (3,  4, BondType.AROMATIC_DOUBLE),
/// ...         (4,  5, BondType.AROMATIC_SINGLE),
/// ...         (5,  0, BondType.AROMATIC_DOUBLE),
/// ...         # Bonds between carbon and hydrogen
/// ...         (0,  6, BondType.SINGLE),
/// ...         (1,  7, BondType.SINGLE),
/// ...         (2,  8, BondType.SINGLE),
/// ...         (3,  9, BondType.SINGLE),
/// ...         (4, 10, BondType.SINGLE),
/// ...         (5, 11, BondType.SINGLE),
/// ...     ])
/// ... )
/// >>> for i, j, bond_type in benzene.bonds.as_array():
/// ...     print(
/// ...         f"{BondType(bond_type).name} bond between "
/// ...         f"{benzene.atom_name[i]} and {benzene.atom_name[j]}"
/// ...     )
/// AROMATIC_SINGLE bond between C1 and C2
/// AROMATIC_DOUBLE bond between C2 and C3
/// AROMATIC_SINGLE bond between C3 and C4
/// AROMATIC_DOUBLE bond between C4 and C5
/// AROMATIC_SINGLE bond between C5 and C6
/// AROMATIC_DOUBLE bond between C1 and C6
/// SINGLE bond between C1 and H1
/// SINGLE bond between C2 and H2
/// SINGLE bond between C3 and H3
/// SINGLE bond between C4 and H4
/// SINGLE bond between C5 and H5
/// SINGLE bond between C6 and H6
///
/// Obtain the bonded atoms for the :math:`C_1`:
///
/// >>> bonds, types = benzene.bonds.get_bonds(0)
/// >>> print(bonds)
/// [1 5 6]
/// >>> print(types)
/// [5 6 1]
/// >>> print(f"C1 is bonded to {', '.join(benzene.atom_name[bonds])}")
/// C1 is bonded to C2, C6, H1
///
/// Cut the benzene molecule in half.
/// Although the first half of the atoms are missing the indices of
/// the cropped :class:`BondList` still represents the bonds of the
/// remaining atoms:
///
/// >>> half_benzene = benzene[
/// ...     np.isin(benzene.atom_name, ["C4", "C5", "C6", "H4", "H5", "H6"])
/// ... ]
/// >>> for i, j, bond_type in half_benzene.bonds.as_array():
/// ...     print(
/// ...         f"{BondType(bond_type).name} bond between "
/// ...         f"{half_benzene.atom_name[i]} and {half_benzene.atom_name[j]}"
/// ...     )
/// AROMATIC_DOUBLE bond between C4 and C5
/// AROMATIC_SINGLE bond between C5 and C6
/// SINGLE bond between C4 and H4
/// SINGLE bond between C5 and H5
/// SINGLE bond between C6 and H6
#[pyclass(module = "biotite.structure", subclass)]
#[derive(Clone)]
pub struct BondList {
    atom_count: usize,
    bonds: Vec<Bond>,
}

#[pymethods]
impl BondList {
    #[new]
    #[pyo3(signature = (atom_count, bonds=None))]
    fn new<'py>(
        py: Python<'py>,
        atom_count: usize,
        bonds: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        match bonds {
            Some(bonds_obj) => {
                // Import numpy and convert to array with int64 dtype
                let np = PyModule::import(py, "numpy")?;
                let array_obj = np.call_method1("asarray", (bonds_obj,))?;

                // Get the shape
                let shape_attr = array_obj.getattr("shape")?;
                let shape: Vec<usize> = shape_attr.extract()?;

                // Validate the shape
                if shape.len() != 2 {
                    return Err(exceptions::PyValueError::new_err(
                        "Expected a 2D-ndarray for input bonds",
                    ));
                }

                let n_bonds = shape[0];
                let n_cols = shape[1];

                if n_cols != 2 && n_cols != 3 {
                    return Err(exceptions::PyValueError::new_err(
                        "Input array containing bonds must be either of shape (n,2) or (n,3)",
                    ));
                }

                let bonds_array: PyReadonlyArray2<isize> = np
                    .call_method1("asarray", (array_obj, np.getattr("intp")?))?
                    .extract()?;
                let array = bonds_array.as_array();

                let mut bond_vec: Vec<Bond> = Vec::with_capacity(n_bonds);
                for i in 0..n_bonds {
                    let bond_type = if n_cols == 3 {
                        let bt_val = array[[i, 2]];
                        if bt_val < 0 {
                            return Err(exceptions::PyValueError::new_err(format!(
                                "BondType {} is invalid",
                                bt_val
                            )));
                        }
                        BondType::try_from(bt_val as u8)?
                    } else {
                        BondType::Any
                    };
                    bond_vec.push(Bond::new(
                        array[[i, 0]],
                        array[[i, 1]],
                        bond_type,
                        atom_count,
                    )?);
                }

                let mut bond_list = BondList {
                    atom_count,
                    bonds: bond_vec,
                };
                bond_list.remove_redundant_bonds();
                Ok(bond_list)
            }
            None => Ok(BondList {
                atom_count,
                bonds: Vec::new(),
            }),
        }
    }

    /// Concatenate multiple :class:`BondList` objects into a single
    /// :class:`BondList`, respectively.
    ///
    /// Parameters
    /// ----------
    /// bonds_lists : iterable object of BondList
    ///     The bond lists to be concatenated.
    ///
    /// Returns
    /// -------
    /// concatenated_bonds : BondList
    ///     The concatenated bond lists.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bonds1 = BondList(2, np.array([(0, 1)]))
    /// >>> bonds2 = BondList(3, np.array([(0, 1), (0, 2)]))
    /// >>> merged_bonds = BondList.concatenate([bonds1, bonds2])
    /// >>> print(merged_bonds.get_atom_count())
    /// 5
    /// >>> print(merged_bonds.as_array()[:, :2])
    /// [[0 1]
    ///  [2 3]
    ///  [2 4]]
    #[staticmethod]
    fn concatenate(bond_lists: Vec<BondList>) -> PyResult<BondList> {
        if bond_lists.is_empty() {
            return Ok(BondList {
                atom_count: 0,
                bonds: Vec::new(),
            });
        }

        let total_bonds: usize = bond_lists.iter().map(|bl| bl.bonds.len()).sum();
        let mut merged_bonds: Vec<Bond> = Vec::with_capacity(total_bonds);
        let mut cum_atom_count: usize = 0;

        for bond_list in &bond_lists {
            for bond in &bond_list.bonds {
                merged_bonds.push(Bond {
                    atom1: bond.atom1 + cum_atom_count,
                    atom2: bond.atom2 + cum_atom_count,
                    bond_type: bond.bond_type,
                });
            }
            cum_atom_count += bond_list.atom_count;
        }

        Ok(BondList {
            atom_count: cum_atom_count,
            bonds: merged_bonds,
        })
    }

    /// offset_indices(offset)
    ///
    /// Increase all atom indices in the :class:`BondList` by the given
    /// offset.
    ///
    /// Implicitly this increases the atom count.
    ///
    /// Parameters
    /// ----------
    /// offset : int
    ///     The atom indices are increased by this value.
    ///     Must be positive.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list = BondList(5, np.array([(1,0),(1,3),(1,4)]))
    /// >>> print(bond_list)
    /// [[0 1 0]
    ///  [1 3 0]
    ///  [1 4 0]]
    /// >>> bond_list.offset_indices(2)
    /// >>> print(bond_list)
    /// [[2 3 0]
    ///  [3 5 0]
    ///  [3 6 0]]
    fn offset_indices(&mut self, offset: isize) -> PyResult<()> {
        if offset < 0 {
            return Err(exceptions::PyValueError::new_err("Offset must be positive"));
        }
        let offset = offset as usize;
        for bond in &mut self.bonds {
            bond.atom1 += offset;
            bond.atom2 += offset;
        }
        self.atom_count += offset;
        Ok(())
    }

    /// as_array()
    ///
    /// Obtain a copy of the internal :class:`ndarray`.
    ///
    /// Returns
    /// -------
    /// array : ndarray, shape=(n,3), dtype=np.uint32
    ///     Copy of the internal :class:`ndarray`.
    ///     For each row, the first column specifies the index of the
    ///     first atom, the second column the index of the second atom
    ///     involved in the bond.
    ///     The third column stores the :class:`BondType`.
    fn as_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let n_bonds = self.bonds.len();
        let mut data: Vec<u32> = Vec::with_capacity(n_bonds * 3);
        for bond in &self.bonds {
            data.push(bond.atom1 as u32);
            data.push(bond.atom2 as u32);
            data.push(bond.bond_type as u32);
        }
        Array2::from_shape_vec((n_bonds, 3), data)
            .expect("Shape mismatch")
            .into_pyarray(py)
    }

    /// Obtain a set representation of the :class:`BondList`.
    ///
    /// Returns
    /// -------
    /// bond_set : set of tuple(int, int, BondType)
    ///     A set of tuples.
    ///     Each tuple represents one bond:
    ///     The first integer represents the first atom,
    ///     the second integer represents the second atom,
    ///     the third value represents the :class:`BondType`.
    fn as_set<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let set = PySet::empty(py)?;
        for bond in &self.bonds {
            let tuple = PyTuple::new(
                py,
                [
                    bond.atom1.into_pyobject(py)?.into_any(),
                    bond.atom2.into_pyobject(py)?.into_any(),
                    bond.bond_type.into_pyobject(py)?.into_any(),
                ],
            )?;
            set.add(tuple)?;
        }
        Ok(set)
    }

    /// as_graph()
    ///
    /// Obtain a graph representation of the :class:`BondList`.
    ///
    /// Returns
    /// -------
    /// bond_set : Graph
    ///     A *NetworkX* :class:`Graph`.
    ///     The atom indices are nodes, the bonds are edges.
    ///     Each edge has a ``"bond_type"`` attribute containing the
    ///     :class:`BondType`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list = BondList(5, np.array([(1,0,2), (1,3,1), (1,4,1)]))
    /// >>> graph = bond_list.as_graph()
    /// >>> print(graph.nodes)
    /// [0, 1, 3, 4]
    /// >>> print(graph.edges)
    /// [(0, 1), (1, 3), (1, 4)]
    /// >>> for i, j in graph.edges:
    /// ...     print(i, j, graph.get_edge_data(i, j))
    /// 0 1 {'bond_type': <BondType.DOUBLE: 2>}
    /// 1 3 {'bond_type': <BondType.SINGLE: 1>}
    /// 1 4 {'bond_type': <BondType.SINGLE: 1>}
    fn as_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let nx = PyModule::import(py, "networkx")?;
        let graph = nx.call_method0("Graph")?;

        let edges = pyo3::types::PyList::empty(py);
        for bond in &self.bonds {
            let edge_data = pyo3::types::PyDict::new(py);
            edge_data.set_item("bond_type", bond.bond_type.into_pyobject(py)?)?;
            let edge_tuple = PyTuple::new(
                py,
                [
                    bond.atom1.into_pyobject(py)?.into_any(),
                    bond.atom2.into_pyobject(py)?.into_any(),
                    edge_data.into_any(),
                ],
            )?;
            edges.append(edge_tuple)?;
        }

        graph.call_method1("add_edges_from", (edges,))?;
        Ok(graph)
    }

    /// Remove aromaticity from the bond types.
    ///
    /// :attr:`BondType.AROMATIC_{ORDER}` is converted into
    /// :attr:`BondType.{ORDER}`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list = BondList(3)
    /// >>> bond_list.add_bond(0, 1, BondType.AROMATIC_SINGLE)
    /// >>> bond_list.add_bond(1, 2, BondType.AROMATIC_DOUBLE)
    /// >>> bond_list.remove_aromaticity()
    /// >>> for i, j, bond_type in bond_list.as_array():
    /// ...     print(i, j, BondType(bond_type).name)
    /// 0 1 SINGLE
    /// 1 2 DOUBLE
    fn remove_aromaticity(&mut self) {
        for bond in &mut self.bonds {
            bond.bond_type = bond.bond_type.without_aromaticity();
        }
    }

    /// Remove the bond order information from aromatic bonds, i.e. convert all
    /// aromatic bonds to :attr:`BondType.ANY`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list = BondList(3)
    /// >>> bond_list.add_bond(0, 1, BondType.AROMATIC_SINGLE)
    /// >>> bond_list.add_bond(1, 2, BondType.AROMATIC_DOUBLE)
    /// >>> bond_list.remove_kekulization()
    /// >>> for i, j, bond_type in bond_list.as_array():
    /// ...     print(i, j, BondType(bond_type).name)
    /// 0 1 AROMATIC
    /// 1 2 AROMATIC
    fn remove_kekulization(&mut self) {
        for bond in &mut self.bonds {
            if matches!(
                bond.bond_type,
                BondType::AromaticSingle | BondType::AromaticDouble | BondType::AromaticTriple
            ) {
                bond.bond_type = BondType::Aromatic;
            }
        }
    }

    /// Convert all bonds to :attr:`BondType.ANY`.
    fn remove_bond_order(&mut self) {
        for bond in &mut self.bonds {
            bond.bond_type = BondType::Any;
        }
    }

    /// convert_bond_type(original_bond_type, new_bond_type)
    ///
    /// Convert all occurences of a given bond type into another bond type.
    ///
    /// Parameters
    /// ----------
    /// original_bond_type : BondType
    ///     The bond type to convert.
    /// new_bond_type : BondType
    ///     The new bond type.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list = BondList(4)
    /// >>> bond_list.add_bond(0, 1, BondType.DOUBLE)
    /// >>> bond_list.add_bond(1, 2, BondType.COORDINATION)
    /// >>> bond_list.add_bond(2, 3, BondType.COORDINATION)
    /// >>> for i, j, bond_type in bond_list.as_array():
    /// ...     print(i, j, BondType(bond_type).name)
    /// 0 1 DOUBLE
    /// 1 2 COORDINATION
    /// 2 3 COORDINATION
    /// >>> bond_list.convert_bond_type(BondType.COORDINATION, BondType.SINGLE)
    /// >>> for i, j, bond_type in bond_list.as_array():
    /// ...     print(i, j, BondType(bond_type).name)
    /// 0 1 DOUBLE
    /// 1 2 SINGLE
    /// 2 3 SINGLE
    fn convert_bond_type(&mut self, original_bond_type: BondType, new_bond_type: BondType) {
        for bond in &mut self.bonds {
            if bond.bond_type == original_bond_type {
                bond.bond_type = new_bond_type;
            }
        }
    }

    /// Get the number atoms represent by this :class:`BondList`.
    ///
    /// Returns
    /// -------
    /// atom_count : int
    ///     The atom count.
    fn get_atom_count(&self) -> usize {
        self.atom_count
    }

    /// Get the total number of bonds.
    ///
    /// Returns
    /// -------
    /// bond_count : int
    ///     The number of bonds. This is equal to the length of the
    ///     internal :class:`ndarray` containing the bonds.
    fn get_bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// get_bonds(atom_index)
    ///
    /// Obtain the indices of the atoms bonded to the atom with the
    /// given index as well as the corresponding bond types.
    ///
    /// Parameters
    /// ----------
    /// atom_index : int
    ///     The index of the atom to get the bonds for.
    ///
    /// Returns
    /// -------
    /// bonds : np.ndarray, dtype=np.uint32, shape=(k,)
    ///     The indices of connected atoms.
    /// bond_types : np.ndarray, dtype=np.uint8, shape=(k,)
    ///     Array of integers, interpreted as :class:`BondType`
    ///     instances.
    ///     This array specifies the type (or order) of the bonds to
    ///     the connected atoms.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list = BondList(5, np.array([(1,0),(1,3),(1,4)]))
    /// >>> bonds, types = bond_list.get_bonds(1)
    /// >>> print(bonds)
    /// [0 3 4]
    #[allow(clippy::type_complexity)]
    fn get_bonds<'py>(
        &self,
        py: Python<'py>,
        atom_index: isize,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u8>>)> {
        let index = to_positive_index(atom_index, self.atom_count)?;

        let mut bonded_atoms: Vec<u32> = Vec::new();
        let mut bond_types: Vec<u8> = Vec::new();

        for bond in &self.bonds {
            if bond.atom1 == index {
                bonded_atoms.push(bond.atom2 as u32);
                bond_types.push(bond.bond_type as u8);
            } else if bond.atom2 == index {
                bonded_atoms.push(bond.atom1 as u32);
                bond_types.push(bond.bond_type as u8);
            }
        }

        Ok((
            Array1::from_vec(bonded_atoms).into_pyarray(py),
            Array1::from_vec(bond_types).into_pyarray(py),
        ))
    }

    /// For each atom index, give the indices of the atoms bonded to
    /// this atom as well as the corresponding bond types.
    ///
    /// Returns
    /// -------
    /// bonds : np.ndarray, dtype=np.int32, shape=(n,k)
    ///     The indices of connected atoms.
    ///     The first dimension represents the atoms,
    ///     the second dimension represents the indices of atoms bonded
    ///     to the respective atom.
    ///     Atoms can have have different numbers of atoms bonded to
    ///     them.
    ///     Therefore, the length of the second dimension *k* is equal
    ///     to the maximum number of bonds for an atom in this
    ///     :class:`BondList`.
    ///     For atoms with less bonds, the corresponding entry in the
    ///     array is padded with ``-1`` values.
    /// bond_types : np.ndarray, dtype=np.int8, shape=(n,k)
    ///     Array of integers, interpreted as :class:`BondType`
    ///     instances.
    ///     This array specifies the bond type (or order) corresponding
    ///     to the returned `bonds`.
    ///     It uses the same ``-1``-padding.
    ///
    /// Examples
    /// --------
    ///
    /// >>> # BondList for benzene
    /// >>> bond_list = BondList(
    /// ...     12,
    /// ...     np.array([
    /// ...         # Bonds between the carbon atoms in the ring
    /// ...         (0,  1, BondType.AROMATIC_SINGLE),
    /// ...         (1,  2, BondType.AROMATIC_DOUBLE),
    /// ...         (2,  3, BondType.AROMATIC_SINGLE),
    /// ...         (3,  4, BondType.AROMATIC_DOUBLE),
    /// ...         (4,  5, BondType.AROMATIC_SINGLE),
    /// ...         (5,  0, BondType.AROMATIC_DOUBLE),
    /// ...         # Bonds between carbon and hydrogen
    /// ...         (0,  6, BondType.SINGLE),
    /// ...         (1,  7, BondType.SINGLE),
    /// ...         (2,  8, BondType.SINGLE),
    /// ...         (3,  9, BondType.SINGLE),
    /// ...         (4, 10, BondType.SINGLE),
    /// ...         (5, 11, BondType.SINGLE),
    /// ...     ])
    /// ... )
    /// >>> bonds, types = bond_list.get_all_bonds()
    /// >>> print(bonds)
    /// [[ 1  5  6 -1]
    ///  [ 0  2  7 -1]
    ///  [ 1  3  8 -1]
    ///  [ 2  4  9 -1]
    ///  [ 3  5 10 -1]
    ///  [ 4  0 11 -1]
    ///  [ 0 -1 -1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 2 -1 -1 -1]
    ///  [ 3 -1 -1 -1]
    ///  [ 4 -1 -1 -1]
    ///  [ 5 -1 -1 -1]]
    /// >>> print(types)
    /// [[ 5  6  1 -1]
    ///  [ 5  6  1 -1]
    ///  [ 6  5  1 -1]
    ///  [ 5  6  1 -1]
    ///  [ 6  5  1 -1]
    ///  [ 5  6  1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 1 -1 -1 -1]]
    /// >>> for i in range(bond_list.get_atom_count()):
    /// ...     bonds_for_atom = bonds[i]
    /// ...     # Remove trailing '-1' values
    /// ...     bonds_for_atom = bonds_for_atom[bonds_for_atom != -1]
    /// ...     print(f"{i}: {bonds_for_atom}")
    /// 0: [1 5 6]
    /// 1: [0 2 7]
    /// 2: [1 3 8]
    /// 3: [2 4 9]
    /// 4: [ 3  5 10]
    /// 5: [ 4  0 11]
    /// 6: [0]
    /// 7: [1]
    /// 8: [2]
    /// 9: [3]
    /// 10: [4]
    /// 11: [5]
    fn get_all_bonds<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i8>>) {
        let n_atoms = self.atom_count;

        // Collect bonds per atom using SmallVec to avoid heap allocation,
        // as usually in chemistry an atom has a maximum of 4 bonds
        let mut per_atom: Vec<SmallVec<[(i32, i8); 4]>> = vec![SmallVec::new(); n_atoms];

        for bond in &self.bonds {
            let bt = bond.bond_type as i8;
            per_atom[bond.atom1].push((bond.atom2 as i32, bt));
            per_atom[bond.atom2].push((bond.atom1 as i32, bt));
        }

        let max_bonds_per_atom = per_atom.iter().map(|v| v.len()).max().unwrap_or(0);
        let mut bonds_matrix = Array2::from_elem((n_atoms, max_bonds_per_atom), -1i32);
        let mut types_matrix = Array2::from_elem((n_atoms, max_bonds_per_atom), -1i8);

        for (i, entries) in per_atom.iter().enumerate() {
            for (j, &(atom_idx, bt)) in entries.iter().enumerate() {
                bonds_matrix[[i, j]] = atom_idx;
                types_matrix[[i, j]] = bt;
            }
        }

        (bonds_matrix.into_pyarray(py), types_matrix.into_pyarray(py))
    }

    /// Represent this :class:`BondList` as adjacency matrix.
    ///
    /// The adjacency matrix is a quadratic matrix with boolean values
    /// according to
    ///
    /// .. math::
    ///
    ///     M_{i,j} =
    ///     \begin{cases}
    ///         \text{True},  & \text{if } \text{Atom}_i \text{ and } \text{Atom}_j \text{ form a bond} \\
    ///         \text{False}, & \text{otherwise}
    ///     \end{cases}.
    ///
    /// Returns
    /// -------
    /// matrix : ndarray, dtype=bool, shape=(n,n)
    ///     The created adjacency matrix.
    ///
    /// Examples
    /// --------
    ///
    /// >>> # BondList for formaldehyde
    /// >>> bond_list = BondList(
    /// ...     4,
    /// ...     np.array([
    /// ...         # Bond between carbon and oxygen
    /// ...         (0,  1, BondType.DOUBLE),
    /// ...         # Bonds between carbon and hydrogen
    /// ...         (0,  2, BondType.SINGLE),
    /// ...         (0,  3, BondType.SINGLE),
    /// ...     ])
    /// ... )
    /// >>> print(bond_list.adjacency_matrix())
    /// [[False  True  True  True]
    ///  [ True False False False]
    ///  [ True False False False]
    ///  [ True False False False]]
    fn adjacency_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        let n = self.atom_count;
        let mut matrix = Array2::from_elem((n, n), false);
        for bond in &self.bonds {
            matrix[[bond.atom1, bond.atom2]] = true;
            matrix[[bond.atom2, bond.atom1]] = true;
        }
        matrix.into_pyarray(py)
    }

    /// Represent this :class:`BondList` as a matrix depicting the bond
    /// type.
    ///
    /// The matrix is a quadratic matrix:
    ///
    /// .. math::
    ///
    ///     M_{i,j} =
    ///     \begin{cases}
    ///         \text{BondType}_{ij},  & \text{if } \text{Atom}_i \text{ and } \text{Atom}_j \text{ form a bond} \\
    ///         -1,                    & \text{otherwise}
    ///     \end{cases}.
    ///
    /// Returns
    /// -------
    /// matrix : ndarray, dtype=np.int8, shape=(n,n)
    ///     The created bond type matrix.
    ///
    /// Examples
    /// --------
    ///
    /// >>> # BondList for formaldehyde
    /// >>> bond_list = BondList(
    /// ...     4,
    /// ...     np.array([
    /// ...         # Bond between carbon and oxygen
    /// ...         (0,  1, BondType.DOUBLE),
    /// ...         # Bonds between carbon and hydrogen
    /// ...         (0,  2, BondType.SINGLE),
    /// ...         (0,  3, BondType.SINGLE),
    /// ...     ])
    /// ... )
    /// >>> print(bond_list.bond_type_matrix())
    /// [[-1  2  1  1]
    ///  [ 2 -1 -1 -1]
    ///  [ 1 -1 -1 -1]
    ///  [ 1 -1 -1 -1]]
    fn bond_type_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i8>> {
        let n = self.atom_count;
        let mut matrix = Array2::from_elem((n, n), -1i8);
        for bond in &self.bonds {
            let bt = bond.bond_type as i8;
            matrix[[bond.atom1, bond.atom2]] = bt;
            matrix[[bond.atom2, bond.atom1]] = bt;
        }
        matrix.into_pyarray(py)
    }

    /// add_bond(atom_index1, atom_index2, bond_type=BondType.ANY)
    ///
    /// Add a bond to the :class:`BondList`.
    ///
    /// If the bond is already existent, only the bond type is updated.
    ///
    /// Parameters
    /// ----------
    /// atom_index1, atom_index2 : int
    ///     The indices of the atoms to create a bond for.
    /// bond_type : BondType
    ///     The type of the bond. Default is :attr:`BondType.ANY`.
    #[pyo3(signature = (atom_index1, atom_index2, bond_type=BondType::Any))]
    fn add_bond(
        &mut self,
        atom_index1: isize,
        atom_index2: isize,
        bond_type: BondType,
    ) -> PyResult<()> {
        let new_bond = Bond::new(atom_index1, atom_index2, bond_type, self.atom_count)?;

        // Check if bond already exists
        for bond in &mut self.bonds {
            if bond.equal_atoms(&new_bond) {
                bond.bond_type = new_bond.bond_type;
                return Ok(());
            }
        }

        // Bond doesn't exist, add it
        self.bonds.push(new_bond);
        Ok(())
    }

    /// remove_bond(atom_index1, atom_index2)
    ///
    /// Remove a bond from the :class:`BondList`.
    ///
    /// If the bond is not existent in the :class:`BondList`, nothing happens.
    ///
    /// Parameters
    /// ----------
    /// atom_index1, atom_index2 : int
    ///     The indices of the atoms whose bond should be removed.
    fn remove_bond(&mut self, atom_index1: isize, atom_index2: isize) -> PyResult<()> {
        let target = Bond::new(atom_index1, atom_index2, BondType::Any, self.atom_count)?;
        self.bonds.retain(|bond| !bond.equal_atoms(&target));
        Ok(())
    }

    /// remove_bonds_to(self, atom_index)
    ///
    /// Remove all bonds from the :class:`BondList` where the given atom
    /// is involved.
    ///
    /// Parameters
    /// ----------
    /// atom_index : int
    ///     The index of the atom whose bonds should be removed.
    fn remove_bonds_to(&mut self, atom_index: isize) -> PyResult<()> {
        let index = to_positive_index(atom_index, self.atom_count)?;
        self.bonds
            .retain(|bond| bond.atom1 != index && bond.atom2 != index);
        Ok(())
    }

    /// remove_bonds(bond_list)
    ///
    /// Remove multiple bonds from the :class:`BondList`.
    ///
    /// All bonds present in `bond_list` are removed from this instance.
    /// If a bond is not existent in this instance, nothing happens.
    /// Only the bond indices, not the bond types, are relevant for
    /// this.
    ///
    /// Parameters
    /// ----------
    /// bond_list : BondList
    ///     The bonds in `bond_list` are removed from this instance.
    fn remove_bonds(&mut self, bond_list: &BondList) {
        let bonds_to_remove: HashSet<(usize, usize)> =
            bond_list.bonds.iter().map(|b| (b.atom1, b.atom2)).collect();
        self.bonds
            .retain(|bond| !bonds_to_remove.contains(&(bond.atom1, bond.atom2)));
    }

    /// merge(bond_list)
    ///
    /// Merge another :class:`BondList` with this instance into a new
    /// object.
    /// If a bond appears in both :class:`BondList`'s, the
    /// :class:`BondType` from the given `bond_list` takes precedence.
    ///
    /// The internal :class:`ndarray` instances containing the bonds are
    /// simply concatenated and the new atom count is the maximum of
    /// both bond lists.
    ///
    /// Parameters
    /// ----------
    /// bond_list : BondList
    ///     This bond list is merged with this instance.
    ///
    /// Returns
    /// -------
    /// bond_list : BondList
    ///     The merged :class:`BondList`.
    ///
    /// Notes
    /// -----
    /// This is not equal to using the `+` operator.
    ///
    /// Examples
    /// --------
    ///
    /// >>> bond_list1 = BondList(3, np.array([(0,1),(1,2)]))
    /// >>> bond_list2 = BondList(5, np.array([(2,3),(3,4)]))
    /// >>> merged_list = bond_list2.merge(bond_list1)
    /// >>> print(merged_list.get_atom_count())
    /// 5
    /// >>> print(merged_list)
    /// [[0 1 0]
    ///  [1 2 0]
    ///  [2 3 0]
    ///  [3 4 0]]
    ///
    /// The BondList given as parameter takes precedence:
    ///
    /// >>> # Specifiy bond type to see where a bond is taken from
    /// >>> bond_list1 = BondList(4, np.array([
    /// ...     (0, 1, BondType.SINGLE),
    /// ...     (1, 2, BondType.SINGLE)
    /// ... ]))
    /// >>> bond_list2 = BondList(4, np.array([
    /// ...     (1, 2, BondType.DOUBLE),    # This one is a duplicate
    /// ...     (2, 3, BondType.DOUBLE)
    /// ... ]))
    /// >>> merged_list = bond_list2.merge(bond_list1)
    /// >>> print(merged_list)
    /// [[0 1 1]
    ///  [1 2 1]
    ///  [2 3 2]]
    fn merge(&self, bond_list: &BondList) -> BondList {
        let new_atom_count = self.atom_count.max(bond_list.atom_count);

        // Concatenate bonds: other bond_list takes precedence (put first)
        let total_bonds = self.bonds.len() + bond_list.bonds.len();
        let mut merged_bonds: Vec<Bond> = Vec::with_capacity(total_bonds);
        merged_bonds.extend(bond_list.bonds.iter().cloned());
        merged_bonds.extend(self.bonds.iter().cloned());

        let mut result = BondList {
            atom_count: new_atom_count,
            bonds: merged_bonds,
        };
        result.remove_redundant_bonds();
        result
    }

    fn __add__(&self, other: &BondList) -> BondList {
        BondList::concatenate(vec![self.clone(), other.clone()]).unwrap()
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Handle single integer index
        if let Ok(int_index) = index.extract::<isize>() {
            let (bonds, types) = self.get_bonds(py, int_index)?;
            let tuple = PyTuple::new(py, [bonds.into_any(), types.into_any()])?;
            return Ok(tuple.into_any());
        }

        // Handle slices
        if let Ok(slice) = index.cast::<pyo3::types::PySlice>() {
            let slice = slice.indices(self.atom_count as isize)?;
            let index_array: Vec<isize> = (0..slice.slicelength as isize)
                .map(|i| slice.start + i * slice.step)
                .collect();
            return self.index_with_int_array(py, &index_array);
        }

        // Convert to numpy array for uniform handling
        let np = PyModule::import(py, "numpy")?;
        let array = np.call_method1("asarray", (index,))?;
        let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;

        if kind == "b" {
            // Boolean mask
            let bool_array: PyReadonlyArray1<bool> = array.extract()?;
            let ndarray_view = bool_array.as_array();
            if ndarray_view.len() != self.atom_count {
                return Err(exceptions::PyIndexError::new_err(format!(
                    "Boolean index has length {}, expected {}",
                    ndarray_view.len(),
                    self.atom_count
                )));
            }
            if let Some(array_view) = ndarray_view.as_slice() {
                let result = self.index_with_bool_mask(array_view)?;
                Ok(result.into_pyobject(py)?.into_any())
            } else {
                // Underlying buffer is not contiguous -> copy into new vector
                let vec: Vec<bool> = ndarray_view.iter().copied().collect();
                let result = self.index_with_bool_mask(&vec)?;
                Ok(result.into_pyobject(py)?.into_any())
            }
        } else {
            // Integer array index
            let int_array: PyReadonlyArray1<isize> = np
                .call_method1("asarray", (&array, np.getattr("intp")?))?
                .extract()?;
            let ndarray_view = int_array.as_array();
            if let Some(array_view) = ndarray_view.as_slice() {
                self.index_with_int_array(py, array_view)
            } else {
                // Underlying buffer is not contiguous -> copy into new vector
                let vec: Vec<isize> = ndarray_view.iter().copied().collect();
                self.index_with_int_array(py, &vec)
            }
        }
    }

    fn __contains__(&self, item: (isize, isize)) -> PyResult<bool> {
        let target = match Bond::new(item.0, item.1, BondType::Any, self.atom_count) {
            Ok(bond) => bond,
            // If the atom indices are out of range, the bond is not part of the bond list
            Err(e) => return Ok(false),
        };
        Ok(self.bonds.iter().any(|bond| bond.equal_atoms(&target)))
    }

    fn __eq__(&self, other: &BondList) -> bool {
        if self.atom_count != other.atom_count {
            return false;
        }
        let self_set: HashSet<&Bond> = self.bonds.iter().collect();
        let other_set: HashSet<&Bond> = other.bonds.iter().collect();
        self_set == other_set
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let array = self.as_array(py);
        array.call_method0("__str__")?.extract()
    }

    fn __iter__(&self) -> PyResult<()> {
        Err(exceptions::PyTypeError::new_err(
            "'BondList' object is not iterable",
        ))
    }

    // Pickling support
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let struc = PyModule::import(py, "biotite.structure")?;
        let cls = struc.getattr("BondList")?;
        let from_state = cls.getattr("_from_state")?;

        // Serialize bonds as flat vector of [atom1, atom2, type] arrays
        let bonds_data: Vec<[usize; 3]> = self
            .bonds
            .iter()
            .map(|b| [b.atom1, b.atom2, b.bond_type as usize])
            .collect();

        let args = PyTuple::new(
            py,
            [
                self.atom_count.into_pyobject(py)?.into_any().unbind(),
                bonds_data.into_pyobject(py)?.into_any().unbind(),
            ],
        )?;

        PyTuple::new(py, [from_state.unbind(), args.into_any().unbind()])
    }

    #[staticmethod]
    #[pyo3(name = "_from_state")]
    fn from_state(atom_count: usize, bonds_data: Vec<[usize; 3]>) -> PyResult<Self> {
        let bonds: Vec<Bond> = bonds_data
            .into_iter()
            .map(|[a1, a2, bt]| {
                Ok(Bond {
                    atom1: a1,
                    atom2: a2,
                    bond_type: BondType::try_from(bt as u8)?,
                })
            })
            .collect::<PyResult<Vec<Bond>>>()?;

        Ok(BondList { atom_count, bonds })
    }

    // Copy support for biotite.Copyable interface
    fn copy(&self) -> BondList {
        self.clone()
    }

    fn __copy__(&self) -> BondList {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> BondList {
        self.clone()
    }
}

impl BondList {
    pub fn get_bonds_ref(&self) -> &[Bond] {
        &self.bonds
    }

    /// Create an empty bond list for internal use from other Rust modules.
    pub fn empty(atom_count: usize) -> Self {
        BondList {
            atom_count,
            bonds: Vec::new(),
        }
    }

    /// Add a bond to the bond list.
    ///
    /// This function is unsafe because it does not check for duplicate bonds.
    /// It is the responsibility of the caller to ensure that the bond is not already in the list
    /// or that ``remove_redundant_bonds()`` is called after adding all bonds.
    pub unsafe fn add(&mut self, bond: Bond) -> PyResult<()> {
        self.bonds.push(bond);
        Ok(())
    }

    /// Remove redundant bonds, keeping the first occurrence.
    /// Uses per-atom neighbor lists instead of HashSet for better performance
    /// with typical chemistry data (max 4 bonds per atom).
    pub fn remove_redundant_bonds(&mut self) {
        let mut seen_per_atom: Vec<SmallVec<[usize; TYPICAL_MAX_BONDS_PER_ATOM]>> =
            vec![SmallVec::new(); self.atom_count];
        self.bonds.retain(|bond| {
            let key = (bond.atom1, bond.atom2);
            if seen.contains(&key) {
                false
            } else {
                seen.insert(key);
                true
            }
        });
    }

    /// Index the bond list with a boolean mask.
    ///
    /// If an atom in a bond is not masked, the bond is removed.
    /// If an atom is masked, its index is decreased by the number of
    /// preceding unmasked atoms (the offset).
    /// The offset is necessary because removing atoms from an AtomArray
    /// decreases the indices of the following atoms.
    fn index_with_bool_mask(&self, mask: &[bool]) -> PyResult<BondList> {
        // Each time an atom is not in the mask, the offset increases by one
        let mut cumulative_offset: usize = 0;
        let offsets: Vec<usize> = mask
            .iter()
            .map(|&is_selected| {
                if !is_selected {
                    cumulative_offset += 1;
                }
                cumulative_offset
            })
            .collect();

        let new_atom_count = mask.iter().filter(|&&x| x).count();

        let mut new_bonds: Vec<Bond> = Vec::new();
        for bond in &self.bonds {
            if mask[bond.atom1] && mask[bond.atom2] {
                // Both atoms involved in bond are masked
                // -> decrease atom index by offset
                new_bonds.push(Bond {
                    atom1: bond.atom1 - offsets[bond.atom1],
                    atom2: bond.atom2 - offsets[bond.atom2],
                    bond_type: bond.bond_type,
                });
            }
            // Otherwise at least one atom is not masked -> bond is dropped
        }

        Ok(BondList {
            atom_count: new_atom_count,
            bonds: new_bonds,
        })
    }

    /// Index the bond list with an integer array.
    ///
    /// Uses an inverse index to efficiently obtain the new index of an
    /// atom, which is especially important for unsorted index arrays.
    /// Bond indices are re-sorted (via `Bond::new`) since the correct
    /// order is not guaranteed for unsorted index arrays.
    fn index_with_int_array<'py>(
        &self,
        py: Python<'py>,
        indices: &[isize],
    ) -> PyResult<Bound<'py, PyAny>> {
        // Build inverse index: if indices[i] = j, then inverse[j] = i.
        // For all atoms j not in the index array, inverse[j] = -1.
        let mut inverse_index: Vec<isize> = vec![-1; self.atom_count];
        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            let pos_idx = to_positive_index(orig_idx, self.atom_count)?;
            if inverse_index[pos_idx] != -1 {
                return Err(exceptions::PyNotImplementedError::new_err(format!(
                    "Duplicate indices are not supported, but index {} appeared multiple times",
                    pos_idx
                )));
            }
            inverse_index[pos_idx] = new_idx as isize;
        }

        let new_atom_count = indices.len();
        let mut new_bonds: Vec<Bond> = Vec::new();

        for bond in &self.bonds {
            let new_idx1 = inverse_index[bond.atom1];
            let new_idx2 = inverse_index[bond.atom2];
            if new_idx1 != -1 && new_idx2 != -1 {
                // Both atoms involved in bond are included by the index array
                // -> assign new atom indices
                new_bonds.push(Bond::new(
                    new_idx1,
                    new_idx2,
                    bond.bond_type,
                    new_atom_count,
                )?);
            }
            // Otherwise at least one atom is not included -> bond is dropped
        }

        let result = BondList {
            atom_count: new_atom_count,
            bonds: new_bonds,
        };
        Ok(result.into_pyobject(py)?.into_any())
    }
}

/// Convert a potentially negative index to a positive index.
#[inline(always)]
fn to_positive_index(index: isize, array_length: usize) -> PyResult<usize> {
    if index < 0 {
        let pos_index = array_length as isize + index;
        if pos_index < 0 {
            return Err(exceptions::PyIndexError::new_err(format!(
                "Index {} is out of range for an atom count of {}",
                index, array_length
            )));
        }
        Ok(pos_index as usize)
    } else {
        if index >= array_length as isize {
            return Err(exceptions::PyIndexError::new_err(format!(
                "Index {} is out of range for an atom count of {}",
                index, array_length
            )));
        }
        Ok(index as usize)
    }
}
