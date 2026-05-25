use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::cmp::max;
use std::collections::{BTreeMap, BTreeSet};

use crate::structure::bonds::{BondList, BondType};
use crate::util::warn;

// Electronegativity of positively charged hydrogen (eV)
const EN_POS_HYDROGEN: f32 = 20.02;

/// Hybridization state of an atom as used in the PEOE algorithm to determine
/// electronegativity parameters:
///   - sp3 (tetrahedral, e.g. C with 4 single bonds)
///   - sp2 (trigonal planar, e.g. C with a double bond, or aromatic C)
///   - sp  (linear, e.g. C with a triple bond)
///
/// The variants are ordered by increasing s-character (sp3 < sp2 < sp),
/// with `None` as the minimum, so that `max()` yields the highest
/// hybridization implied by any of an atom's bonds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Hybridization {
    /// No hybridization as the atom is not part of a molecule (e.g. an ion)
    None,
    /// No hybridization could be determined (unsupported element or bond cases)
    Invalid,
    Sp3,
    Sp2,
    Sp,
}

/// Result of looking up electronegativity parameters for an element and
/// hybridization state.
enum LookupResult {
    /// Parameters found for this element and hybridization.
    Found(ElectronegativityParameters),
    /// Element is known but has no parameters for this hybridization.
    UnknownHybridization,
    /// Element is not in the parameter table at all.
    UnknownElement,
}

/// Electronegativity parameters for a single atom as used in the PEOE
/// charge transfer equation.
///
/// `NaN` values indicate that no parameters are available for this atom.
#[derive(Clone, Copy)]
struct ElectronegativityParameters {
    /// Coefficient for the constant term.
    a: f32,
    /// Coefficient for the linear term.
    b: f32,
    /// Coefficient for the quadratic term.
    c: f32,
    /// Electronegativity at charge +1, used as divisor in charge transfer.
    /// For hydrogen, the special value `EN_POS_HYDROGEN` is used.
    pos_en: f32,
}

impl Default for ElectronegativityParameters {
    fn default() -> Self {
        Self {
            a: f32::NAN,
            b: f32::NAN,
            c: f32::NAN,
            pos_en: f32::NAN,
        }
    }
}

impl ElectronegativityParameters {
    /// Compute the electronegativity for the given charge as
    ///
    /// .. math::
    ///
    ///     \chi = a + bQ + cQ^2.
    #[inline(always)]
    fn electronegativity(&self, charge: f32) -> f32 {
        self.a + self.b * charge + self.c * charge * charge
    }
}

/// Determine the hybridization state for each atom from the BondList.
///
/// The hybridization is determined by the highest bond order among an
/// atom's bonds:
///   - Any aromatic bond -> sp2 (atoms in aromatic rings are always sp2)
///   - Triple bond -> sp
///   - Double bond -> sp2
///   - Single bond -> sp3
///   - Any/Coordination bond -> `Invalid` (does not contribute info)
///
/// Atoms that remain `Invalid` after processing all bonds (i.e. all
/// their bonds are `Any` or `Coordination`) are resolved via
/// `hybridization_from_partner_count` as a fallback.
/// A `UserWarning` is raised for these atoms.
fn get_hybridization(
    py: Python<'_>,
    bond_list: &BondList,
    elements: &[String],
) -> PyResult<Vec<Hybridization>> {
    let n_atoms = bond_list.get_atom_count();
    let bonds = bond_list.get_bonds_ref();

    let mut n_binding_partners: Vec<usize> = vec![0; n_atoms];
    let mut hybridizations: Vec<Hybridization> = vec![Hybridization::None; n_atoms];

    for bond in bonds {
        let atom1 = bond.atom1;
        let atom2 = bond.atom2;
        n_binding_partners[atom1] += 1;
        n_binding_partners[atom2] += 1;

        let implied_hybridization = match bond.bond_type {
            // Any/Coordination bonds do not provide hybridization information
            BondType::Any | BondType::Coordination => Hybridization::Invalid,
            BondType::Single => Hybridization::Sp3,
            BondType::Double => Hybridization::Sp2,
            BondType::Triple | BondType::Quadruple => Hybridization::Sp,
            // Atoms participating in any aromatic bond are sp2 hybridized
            BondType::AromaticSingle
            | BondType::AromaticDouble
            | BondType::AromaticTriple
            | BondType::Aromatic => Hybridization::Sp2,
        };

        hybridizations[atom1] = max(hybridizations[atom1], implied_hybridization);
        hybridizations[atom2] = max(hybridizations[atom2], implied_hybridization);
    }

    // For atoms that remain `Invalid` (all bonds are Any/Coordination),
    // fall back to inferring hybridization from the number of binding partners
    let mut atoms_without_bond_type: Vec<usize> = Vec::new();
    for atom_index in 0..n_atoms {
        if hybridizations[atom_index] == Hybridization::Invalid {
            atoms_without_bond_type.push(atom_index);
            hybridizations[atom_index] = hybridization_from_partner_count(
                &elements[atom_index],
                n_binding_partners[atom_index],
            );
        }
    }

    // Warn about atoms whose hybridization was inferred from partner count
    let n_without_bond_type = atoms_without_bond_type.len();
    if n_without_bond_type == n_atoms {
        warn::<exceptions::PyUserWarning>(
            py,
            "Each atom's bond type is undefined. Therefore, it is \
             resorted to the amount of binding partners for the \
             identification of the hybridization state which can lead \
             to erroneous results.",
        )?;
    } else if n_without_bond_type > 0 {
        let indices_str: String = atoms_without_bond_type
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        warn::<exceptions::PyUserWarning>(
            py,
            &format!(
                "Some atoms' bond type is unspecified, i. e. the bond \
                 type is given as `any`. For these atoms, identification \
                 of the hybridization state is performed via the amount \
                 of binding partners which can lead to erroneous results.\
                 \n\n\
                 In detail, these atoms possess the following indices: \n\
                 {indices_str}."
            ),
        )?;
    }

    Ok(hybridizations)
}

/// Infer hybridization from the number of binding partners.
///
/// This is used as a fallback when bond types are not available
/// (all bonds are `BondType::Any`).
/// Can lead to erroneous results, e.g. for charged atoms where the
/// partner count does not reflect the true hybridization state.
#[inline(always)]
fn hybridization_from_partner_count(element: &str, n_partners: usize) -> Hybridization {
    match element {
        "H" => match n_partners {
            1 => Hybridization::Sp3,
            _ => Hybridization::None,
        },
        "C" => match n_partners {
            4 => Hybridization::Sp3,
            3 => Hybridization::Sp2,
            2 => Hybridization::Sp,
            _ => Hybridization::None,
        },
        "N" => match n_partners {
            3 | 4 => Hybridization::Sp3,
            2 => Hybridization::Sp2,
            1 => Hybridization::Sp,
            _ => Hybridization::None,
        },
        "O" => match n_partners {
            2 => Hybridization::Sp3,
            1 => Hybridization::Sp2,
            _ => Hybridization::None,
        },
        "S" => match n_partners {
            2 => Hybridization::Sp3,
            _ => Hybridization::None,
        },
        "F" => match n_partners {
            1 => Hybridization::Sp3,
            _ => Hybridization::None,
        },
        "Cl" => match n_partners {
            1 => Hybridization::Sp3,
            _ => Hybridization::None,
        },
        "Br" => match n_partners {
            1 => Hybridization::Sp3,
            _ => Hybridization::None,
        },
        "I" => match n_partners {
            1 => Hybridization::Sp3,
            _ => Hybridization::None,
        },
        _ => Hybridization::None,
    }
}

/// Look up electronegativity parameters for a given element and
/// hybridization state.
fn lookup_electronegativity(element: &str, hybridization: Hybridization) -> LookupResult {
    let (a, b, c) = match element {
        "H" => match hybridization {
            // Hydrogen is always treated as s-orbital
            Hybridization::Sp3 => (7.17, 6.24, -0.56),
            _ => return LookupResult::UnknownHybridization,
        },
        "C" => match hybridization {
            Hybridization::Sp3 => (7.98, 9.18, 1.88),
            Hybridization::Sp2 => (8.79, 9.18, 1.88),
            Hybridization::Sp => (10.39, 9.45, 0.73),
            _ => return LookupResult::UnknownHybridization,
        },
        "N" => match hybridization {
            Hybridization::Sp3 => (11.54, 10.82, 1.36),
            Hybridization::Sp2 => (12.87, 11.15, 0.85),
            Hybridization::Sp => (15.68, 11.7, -0.27),
            _ => return LookupResult::UnknownHybridization,
        },
        "O" => match hybridization {
            Hybridization::Sp3 => (14.18, 12.92, 1.39),
            Hybridization::Sp2 => (17.07, 13.79, 0.47),
            _ => return LookupResult::UnknownHybridization,
        },
        "S" => match hybridization {
            Hybridization::Sp3 => (10.14, 9.13, 1.38),
            _ => return LookupResult::UnknownHybridization,
        },
        "F" => match hybridization {
            Hybridization::Sp3 => (14.66, 13.85, 2.31),
            _ => return LookupResult::UnknownHybridization,
        },
        "Cl" => match hybridization {
            Hybridization::Sp3 => (11.00, 9.69, 1.35),
            _ => return LookupResult::UnknownHybridization,
        },
        "Br" => match hybridization {
            Hybridization::Sp3 => (10.08, 8.47, 1.16),
            _ => return LookupResult::UnknownHybridization,
        },
        "I" => match hybridization {
            Hybridization::Sp3 => (9.90, 7.96, 0.96),
            _ => return LookupResult::UnknownHybridization,
        },
        _ => return LookupResult::UnknownElement,
    };
    let pos_en = if is_heavy(element) {
        a + b + c
    } else {
        EN_POS_HYDROGEN
    };
    LookupResult::Found(ElectronegativityParameters { a, b, c, pos_en })
}

/// Check if an element is heavy (not hydrogen).
#[inline(always)]
fn is_heavy(element: &str) -> bool {
    element != "H" && element != "D"
}

/// Look up electronegativity parameters for each atom based on its element
/// and hybridization state.
fn get_en_parameters(
    py: Python<'_>,
    elements: &[String],
    hybridizations: &[Hybridization],
) -> PyResult<Vec<ElectronegativityParameters>> {
    let n_atoms = elements.len();
    let mut params: Vec<ElectronegativityParameters> =
        vec![ElectronegativityParameters::default(); n_atoms];

    // Collect unparametrized cases for warnings (deduplicated)
    let mut unknown_elements: BTreeSet<String> = BTreeSet::new();
    let mut unknown_valences: BTreeMap<(String, Hybridization), ()> = BTreeMap::new();

    for atom_index in 0..n_atoms {
        let element = &elements[atom_index];
        let hybridization = hybridizations[atom_index];

        if hybridization == Hybridization::None {
            // Ions: no bonds -> keep NaN parameters (formal charge used directly)
            continue;
        }

        match lookup_electronegativity(element, hybridization) {
            LookupResult::Found(en_params) => {
                params[atom_index] = en_params;
            }
            LookupResult::UnknownElement => {
                unknown_elements.insert(element.clone());
            }
            LookupResult::UnknownHybridization => {
                unknown_valences.insert((element.clone(), hybridization), ());
            }
        }
    }

    // Warn about unknown valence states
    if !unknown_valences.is_empty() {
        let entries: Vec<String> = unknown_valences
            .keys()
            .map(|(elem, hyb)| format!("{:<10}{:?}", elem, hyb))
            .collect();
        warn::<exceptions::PyUserWarning>(
            py,
            &format!(
                "Parameters for specific valence states of some atoms \
                 are not available. These valence states are: \n\
                 Atom:     Hybridization:\n\
                 {}\n\
                 Their electronegativity is given as NaN.",
                entries.join("\n")
            ),
        )?;
    }

    // Warn about completely unknown elements
    if !unknown_elements.is_empty() {
        let elements_str: String = unknown_elements
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        warn::<exceptions::PyUserWarning>(
            py,
            &format!(
                "Parameters required for computation of \
                 electronegativity aren't available for the following \
                 atoms: {elements_str}. \
                 Their electronegativity is given as NaN."
            ),
        )?;
    }

    Ok(params)
}

/// Compute PEOE partial charges.
///
/// This performs hybridization determination, parameter lookup,
/// and the iterative charge transfer loop.
///
/// Parameters
/// ----------
/// elements
///     Element symbols for each atom.
/// charges
///     Initial charges (formal charges) for each atom, shape (n,).
/// bond_list
///     The BondList describing the molecular connectivity.
/// iteration_step_num
///     Number of PEOE iteration steps.
///
/// Returns
/// -------
/// charges
///     The computed partial charges.
#[pyfunction]
pub fn partial_charges<'py>(
    py: Python<'py>,
    elements: Vec<String>,
    charges: PyReadonlyArray1<'py, f32>,
    bond_list: &BondList,
    iteration_step_num: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let mut charges: Vec<f32> = charges.as_slice()?.to_vec();
    let n_atoms = charges.len();

    // Look up electronegativity parameters for each atom and warn about missing ones
    let en_params =
        get_en_parameters(py, &elements, &get_hybridization(py, bond_list, &elements)?)?;

    // Pre-allocate buffer for electronegativity values of each atom
    let mut en_values: Vec<f32> = vec![0.0; n_atoms];
    let mut damping: f32 = 1.0;
    for _ in 0..iteration_step_num {
        // In the beginning of each iteration step, the damping factor is
        // halved in order to guarantee rapid convergence
        damping *= 0.5;

        for atom_index in 0..n_atoms {
            en_values[atom_index] = en_params[atom_index].electronegativity(charges[atom_index]);
        }

        // Iterate over bonds to transfer charges
        // based on new electronegativity values
        for bond in bond_list.get_bonds_ref().iter() {
            let atom_i = bond.atom1;
            let atom_j = bond.atom2;

            let en_i = en_values[atom_i];
            let en_j = en_values[atom_j];
            // For atoms that are not available in the parameter tables,
            // but which are incorporated into molecules,
            // the partial charge is set to NaN
            if en_i.is_nan() || en_j.is_nan() {
                // Determining for which atom exactly no parameters are
                // available is necessary since the other atom, for which
                // there indeed are parameters, could be involved in
                // multiple bonds
                if en_i.is_nan() {
                    charges[atom_i] = f32::NAN;
                }
                if en_j.is_nan() {
                    charges[atom_j] = f32::NAN;
                }
            } else {
                let divisor = if en_j > en_i {
                    en_params[atom_i].pos_en
                } else {
                    en_params[atom_j].pos_en
                };
                let charge_transfer = ((en_j - en_i) / divisor) * damping;
                charges[atom_i] += charge_transfer;
                charges[atom_j] -= charge_transfer;
            }
        }
        py.check_signals()?;
    }

    Ok(charges.into_pyarray(py))
}
