use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::f32::consts::PI;

use crate::structure::celllist::{CellList, Radius};
use crate::structure::util::{distance_squared, extract_coord};
use crate::util::check_signals_periodically;

/// Calculate the Solvent Accessible Surface Area (SASA) using the Shrake-Rupley algorithm.
///
/// Parameters
/// ----------
/// coord
///     Coordinates of all atoms.
/// radii
///     VdW radii + probe radius for each atom.
/// sphere_points
///     Points on the surface of a unit sphere.
/// sasa_filter
///     Boolean mask indicating which atoms to calculate SASA for.
/// occlusion_filter
///     Boolean mask indicating which atoms are considered for occlusion.
///
/// Returns
/// -------
/// sasa
///     Atom-wise SASA. NaN for atoms where SASA was not calculated.
#[pyfunction]
pub fn sasa<'py>(
    py: Python<'py>,
    coord: PyReadonlyArray2<'py, f32>,
    radii: PyReadonlyArray1<'py, f32>,
    sphere_points: PyReadonlyArray2<'py, f32>,
    sasa_filter: PyReadonlyArray1<'py, bool>,
    occlusion_filter: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let coord = extract_coord(coord)?;
    let radii = radii.as_slice()?;
    let sphere_points = extract_coord(sphere_points)?;
    let sasa_filter = sasa_filter.as_slice()?;
    let occlusion_filter = occlusion_filter.as_slice()?;

    let n_atoms = coord.len();
    let n_sphere_points = sphere_points.len();
    // Area of a single point on a unit sphere
    let area_per_point = 4.0 * PI / (n_sphere_points as f32);

    if radii.len() != n_atoms {
        return Err(exceptions::PyIndexError::new_err(format!(
            "{} radii were given for {} atoms",
            radii.len(),
            n_atoms
        )));
    }
    if sasa_filter.len() != n_atoms {
        return Err(exceptions::PyValueError::new_err(format!(
            "Mask has length {}, but {} atoms were provided",
            sasa_filter.len(),
            n_atoms
        )));
    }
    if occlusion_filter.len() != n_atoms {
        return Err(exceptions::PyValueError::new_err(format!(
            "Mask has length {}, but {} atoms were provided",
            occlusion_filter.len(),
            n_atoms
        )));
    }

    // Filter coordinates and radii for occluding atoms
    // Also track the original index for each occluding atom
    let mut occlusion_coord: Vec<[f32; 3]> = Vec::new();
    let mut occlusion_radii: Vec<f32> = Vec::new();
    let mut occlusion_orig_idx: Vec<usize> = Vec::new();
    for i in 0..n_atoms {
        if occlusion_filter[i] {
            occlusion_coord.push(coord[i]);
            occlusion_radii.push(radii[i]);
            occlusion_orig_idx.push(i);
        }
    }
    if occlusion_coord.is_empty() {
        return Err(exceptions::PyValueError::new_err(
            "No atoms are within the occlusion filter",
        ));
    }
    // Pre-compute squared radii for occluding atoms
    let occlusion_radii_sq: Vec<f32> = occlusion_radii.iter().map(|r| r * r).collect();

    // Cell size is as large as the maximum distance where two atoms can intersect.
    // Therefore intersecting atoms are always in the same or adjacent cell.
    let max_occlusion_radius = occlusion_radii.iter().cloned().fold(0.0f32, f32::max);
    let cell_size = max_occlusion_radius * 2.0;
    let cell_list = CellList::new(
        occlusion_coord.clone(),
        cell_size,
        None,
        None,
        occlusion_coord.len(),
    )?;

    // Initialize result with NaN
    let mut sasa_result: Vec<f32> = vec![f32::NAN; n_atoms];
    // Buffer for storing relevant occluding atoms, reused across iterations
    // Each entry is (coord, radius_sq)
    let mut relevant_occluders: Vec<([f32; 3], f32)> = Vec::new();
    for i in 0..n_atoms {
        if !sasa_filter[i] {
            continue;
        }

        let atom_coord = coord[i];
        let atom_radius = radii[i];
        let atom_radius_sq = atom_radius * atom_radius;

        // Find adjacent atoms from cell list
        // Query with a single coordinate
        // Search radius must cover all potentially intersecting atoms:
        // Two spheres intersect if distance < radius1 + radius2
        // So we need atoms within atom_radius + max_occlusion_radius
        let adjacent_pairs = cell_list.get_atoms_from_slice(
            py,
            &[atom_coord],
            &Radius::Single(atom_radius + max_occlusion_radius),
        )?;

        // Filter adjacent atoms to only those that actually intersect with the current atom
        relevant_occluders.clear();
        for [_, adjacent_idx] in &adjacent_pairs {
            // Skip if this is the same atom we're computing SASA for
            if occlusion_orig_idx[*adjacent_idx] == i {
                continue;
            }

            let occlusion_atom_coord = occlusion_coord[*adjacent_idx];
            let adjacent_radius = occlusion_radii[*adjacent_idx];
            let dist_sq = distance_squared(atom_coord, occlusion_atom_coord);
            // Only include if spheres intersect
            if dist_sq < (adjacent_radius + atom_radius).powi(2) {
                relevant_occluders.push((occlusion_atom_coord, occlusion_radii_sq[*adjacent_idx]));
            }
        }

        // Count accessible sphere points
        let mut n_accessible = n_sphere_points;
        for sphere_point in &sphere_points {
            // Transform unit sphere point to the atom's sphere
            let point = [
                sphere_point[0] * atom_radius + atom_coord[0],
                sphere_point[1] * atom_radius + atom_coord[1],
                sphere_point[2] * atom_radius + atom_coord[2],
            ];

            // Check if this point is occluded by any neighboring atom
            for (occlusion_coord, occlusion_radius_sq) in &relevant_occluders {
                let dist_sq = distance_squared(point, *occlusion_coord);
                if dist_sq < *occlusion_radius_sq {
                    // Point is inside the occluding atom's sphere
                    n_accessible -= 1;
                    break;
                }
            }
        }

        // Calculate SASA for this atom
        sasa_result[i] = area_per_point * (n_accessible as f32) * atom_radius_sq;

        check_signals_periodically(py, i)?;
    }

    Ok(sasa_result.into_pyarray(py))
}
