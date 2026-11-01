use numpy::ndarray::Array;
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions;
use pyo3::prelude::*;

/// Maximum number of Newton-Raphson iterations to find the largest eigenvalue.
const MAX_NEWTON_ITERATIONS: usize = 50;
/// Newton-Raphson stops, if the eigenvalue changes less than this value,
/// relative to its initial guess.
const EIGENVALUE_PRECISION: f64 = 1e-12;
/// If the chosen column of the adjoint matrix, relative to the cubed magnitude of the
/// key matrix, is smaller than this value, the largest eigenvalue is (nearly) degenerate
/// and the eigenvector cannot be computed reliably from the adjoint matrix.
const ADJOINT_PRECISION: f64 = 1e-6;
/// Maximum number of sweeps in the Jacobi eigenvalue algorithm.
const MAX_JACOBI_SWEEPS: usize = 50;

/// Find the rigid transformations that superimpose the `mobile` structures onto the
/// `fixed` structures with minimum RMSD.
///
/// The optimal rotation is obtained via the quaternion-based characteristic
/// polynomial (QCP) method:
/// The largest eigenvalue of the key matrix is found by Newton-Raphson iteration
/// on its characteristic polynomial (Theobald, Acta Cryst A, 61, 2005) and the
/// corresponding eigenvector, i.e. the quaternion representing the rotation, is
/// obtained from the adjoint matrix (Liu et al., J Comput Chem, 31, 2010).
///
/// Parameters
/// ----------
/// fixed, mobile
///     The C-contiguous coordinates to be superimposed, with shape *(m, n, 3)*.
///     One of them may also have a single model, which is then used for all models
///     of the other one.
/// atom_mask
///     If given, only the atoms covered by this boolean mask are considered for
///     superimposition.
///
/// Returns
/// -------
/// rotation
///     The rotation matrices with shape *(m, 3, 3)*.
/// translation
///     The translation vectors with shape *(m, 3)*, that are applied after the rotation.
#[pyfunction]
#[pyo3(signature = (fixed, mobile, atom_mask=None))]
#[allow(clippy::type_complexity)]
pub fn superimpose<'py>(
    py: Python<'py>,
    fixed: PyReadonlyArray3<'py, f32>,
    mobile: PyReadonlyArray3<'py, f32>,
    atom_mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<f64>>)> {
    let fixed_shape = fixed.shape();
    let mobile_shape = mobile.shape();
    if fixed_shape[2] != 3 || mobile_shape[2] != 3 {
        return Err(exceptions::PyValueError::new_err(
            "Coordinates must have form (x,y,z)",
        ));
    }
    let n_atoms = fixed_shape[1];
    if mobile_shape[1] != n_atoms {
        return Err(exceptions::PyIndexError::new_err(format!(
            "Fixed structure has {} atoms, but mobile structure has {} atoms",
            n_atoms, mobile_shape[1]
        )));
    }
    let atom_mask = match &atom_mask {
        Some(mask) => Some(mask.as_slice()?),
        None => None,
    };
    if let Some(mask) = atom_mask {
        if mask.len() != n_atoms {
            return Err(exceptions::PyIndexError::new_err(format!(
                "Mask has length {}, but {} atoms were provided",
                mask.len(),
                n_atoms
            )));
        }
    }
    let n_fixed_models = fixed_shape[0];
    let n_mobile_models = mobile_shape[0];
    let n_models = n_fixed_models.max(n_mobile_models);
    if (n_fixed_models != n_models && n_fixed_models != 1)
        || (n_mobile_models != n_models && n_mobile_models != 1)
    {
        return Err(exceptions::PyIndexError::new_err(format!(
            "Fixed structure has {} models, but mobile structure has {} models",
            n_fixed_models, n_mobile_models
        )));
    }
    let fixed = fixed.as_slice()?;
    let mobile = mobile.as_slice()?;

    let model_size = n_atoms * 3;
    let mut rotations: Vec<f64> = Vec::with_capacity(n_models * 9);
    let mut translations: Vec<f64> = Vec::with_capacity(n_models * 3);
    for m in 0..n_models {
        // A single model is used for all models of the other structure
        let fixed_model = if n_fixed_models == 1 { 0 } else { m };
        let mobile_model = if n_mobile_models == 1 { 0 } else { m };
        let (rotation, translation) = superimpose_model(
            &fixed[fixed_model * model_size..(fixed_model + 1) * model_size],
            &mobile[mobile_model * model_size..(mobile_model + 1) * model_size],
            atom_mask,
        )?;
        rotations.extend(rotation.iter().flatten());
        translations.extend(translation);
    }

    let rotations =
        Array::from_shape_vec((n_models, 3, 3), rotations).expect("Vector size matches the shape");
    let translations =
        Array::from_shape_vec((n_models, 3), translations).expect("Vector size matches the shape");
    Ok((rotations.into_pyarray(py), translations.into_pyarray(py)))
}

/// Find the rotation matrix and translation vector for a single model.
fn superimpose_model(
    fixed: &[f32],
    mobile: &[f32],
    atom_mask: Option<&[bool]>,
) -> PyResult<([[f64; 3]; 3], [f64; 3])> {
    // First pass: Compute centroids
    let mut fixed_centroid = [0.0; 3];
    let mut mobile_centroid = [0.0; 3];
    let mut n_selected: usize = 0;
    for_each_selected(fixed, mobile, atom_mask, |fixed_coord, mobile_coord| {
        for k in 0..3 {
            fixed_centroid[k] += fixed_coord[k];
            mobile_centroid[k] += mobile_coord[k];
        }
        n_selected += 1;
    });
    if n_selected == 0 {
        return Err(exceptions::PyValueError::new_err(
            "No atoms are selected for superimposition",
        ));
    }
    for k in 0..3 {
        fixed_centroid[k] /= n_selected as f64;
        mobile_centroid[k] /= n_selected as f64;
    }

    // Second pass: Compute the inner product matrix 'S' of the centered coordinates
    // and the inner products of each structure with itself
    let mut s = [[0.0; 3]; 3];
    let mut fixed_inner_product = 0.0;
    let mut mobile_inner_product = 0.0;
    for_each_selected(fixed, mobile, atom_mask, |fixed_coord, mobile_coord| {
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        for k in 0..3 {
            a[k] = fixed_coord[k] - fixed_centroid[k];
            b[k] = mobile_coord[k] - mobile_centroid[k];
        }
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += b[i] * a[j];
            }
        }
        fixed_inner_product += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        mobile_inner_product += b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    });

    let rotation = optimal_rotation(&s, fixed_inner_product, mobile_inner_product);
    // The mobile centroid is moved to the origin, the rotation is applied
    // and the coordinates are moved to the fixed centroid
    let mut translation = fixed_centroid;
    for i in 0..3 {
        for j in 0..3 {
            translation[i] -= rotation[i][j] * mobile_centroid[j];
        }
    }
    Ok((rotation, translation))
}

/// Call `f` for each pair of fixed and mobile coordinates selected by the mask.
#[inline(always)]
fn for_each_selected<F: FnMut([f64; 3], [f64; 3])>(
    fixed: &[f32],
    mobile: &[f32],
    atom_mask: Option<&[bool]>,
    mut f: F,
) {
    let to_f64 = |c: &[f32]| [c[0] as f64, c[1] as f64, c[2] as f64];
    let pairs = fixed.chunks_exact(3).zip(mobile.chunks_exact(3));
    // Separate loops allow the compiler to optimize the unmasked case
    match atom_mask {
        None => pairs.for_each(|(a, b)| f(to_f64(a), to_f64(b))),
        Some(mask) => pairs.zip(mask).for_each(|((a, b), &selected)| {
            if selected {
                f(to_f64(a), to_f64(b))
            }
        }),
    }
}

/// Find the rotation matrix that minimizes the RMSD between centered structures *A*
/// (fixed) and *B* (mobile).
///
/// `s` is the inner product matrix :math:`S_{ij} = \sum_k B_{ki} A_{kj}`,
/// `inner_product_a` and `inner_product_b` are the sums of squared coordinates of
/// *A* and *B*, respectively.
fn optimal_rotation(
    s: &[[f64; 3]; 3],
    inner_product_a: f64,
    inner_product_b: f64,
) -> [[f64; 3]; 3] {
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = *s;
    // The key matrix, whose eigenvector belonging to the largest eigenvalue
    // is the quaternion representing the optimal rotation
    let key = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];

    // The largest eigenvalue is bounded from above by this value,
    // as the minimum squared deviation (G_A + G_B - 2 * lambda_max) cannot be negative
    let upper_bound = (inner_product_a + inner_product_b) / 2.0;
    if upper_bound == 0.0 {
        // All selected atoms are at the same position in both structures
        // -> any rotation is optimal
        return IDENTITY;
    }

    // As the key matrix is traceless, the characteristic polynomial is
    // lambda^4 + c2 * lambda^2 + c1 * lambda + c0
    let c2 = -2.0 * s.iter().flatten().map(|x| x * x).sum::<f64>();
    let c1 = -8.0 * determinant_3x3(s);
    let (_, c0) = adjoint_and_determinant_4x4(&key);
    // Find the largest root of the characteristic polynomial via Newton-Raphson,
    // starting from the upper bound
    let mut eigenvalue = upper_bound;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        let eigenvalue_sq = eigenvalue * eigenvalue;
        let value = (eigenvalue_sq + c2) * eigenvalue_sq + c1 * eigenvalue + c0;
        let derivative = (4.0 * eigenvalue_sq + 2.0 * c2) * eigenvalue + c1;
        if derivative == 0.0 {
            break;
        }
        let delta = value / derivative;
        eigenvalue -= delta;
        if delta.abs() <= EIGENVALUE_PRECISION * upper_bound {
            break;
        }
    }

    // For the eigenvalue, the matrix (K - lambda * I) is singular,
    // so each nonzero column of its adjoint matrix is an eigenvector
    let mut shifted_key = key;
    for (i, row) in shifted_key.iter_mut().enumerate() {
        row[i] -= eigenvalue;
    }
    let (adjoint, _) = adjoint_and_determinant_4x4(&shifted_key);
    // The adjoint matrix is (approximately) proportional to the outer product of the
    // eigenvector with itself
    // -> the column with the largest diagonal element is the most accurate one
    let mut best_column = 0;
    for j in 1..4 {
        if adjoint[j][j].abs() > adjoint[best_column][best_column].abs() {
            best_column = j;
        }
    }
    let quaternion =
        if adjoint[best_column][best_column].abs() > ADJOINT_PRECISION * upper_bound.powi(3) {
            [
                adjoint[0][best_column],
                adjoint[1][best_column],
                adjoint[2][best_column],
                adjoint[3][best_column],
            ]
        } else {
            // The largest eigenvalue is (nearly) degenerate, e.g. for collinear atoms
            // -> the adjoint matrix vanishes and the eigenvector is computed
            // via a full eigendecomposition instead
            largest_eigenvector(key)
        };
    quaternion_to_rotation(quaternion)
}

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn determinant_3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Compute the adjoint matrix and the determinant of a 4x4 matrix
/// via its 2x2 minors.
fn adjoint_and_determinant_4x4(m: &[[f64; 4]; 4]) -> ([[f64; 4]; 4], f64) {
    // 2x2 minors of the upper two rows
    let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
    // 2x2 minors of the lower two rows
    let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    let determinant = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    let adjoint = [
        [
            m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3,
            -m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3,
            m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3,
            -m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3,
        ],
        [
            -m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1,
            m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1,
            -m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1,
            m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1,
        ],
        [
            m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0,
            -m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0,
            m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0,
            -m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0,
        ],
        [
            -m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0,
            m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0,
            -m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0,
            m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0,
        ],
    ];
    (adjoint, determinant)
}

/// Index pairs of the upper triangle (without diagonal) of a 4x4 matrix.
const UPPER_TRIANGLE: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Find the eigenvector belonging to the largest eigenvalue of a symmetric 4x4
/// matrix via the cyclic Jacobi eigenvalue algorithm.
fn largest_eigenvector(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut eigenvectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..MAX_JACOBI_SWEEPS {
        let total = a.iter().flatten().map(|x| x * x).sum::<f64>();
        let off_diagonal = UPPER_TRIANGLE
            .iter()
            .map(|&(p, q)| a[p][q] * a[p][q])
            .sum::<f64>();
        if off_diagonal <= f64::EPSILON * f64::EPSILON * total {
            break;
        }
        for &(p, q) in UPPER_TRIANGLE.iter() {
            if a[p][q] == 0.0 {
                continue;
            }
            // Rotate in the (p, q) plane to eliminate the element at (p, q)
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let cos = 1.0 / (t * t + 1.0).sqrt();
            let sin = t * cos;
            for row in a.iter_mut() {
                let (akp, akq) = (row[p], row[q]);
                row[p] = cos * akp - sin * akq;
                row[q] = sin * akp + cos * akq;
            }
            let (row_p, row_q) = (a[p], a[q]);
            for k in 0..4 {
                a[p][k] = cos * row_p[k] - sin * row_q[k];
                a[q][k] = sin * row_p[k] + cos * row_q[k];
            }
            for row in eigenvectors.iter_mut() {
                let (vkp, vkq) = (row[p], row[q]);
                row[p] = cos * vkp - sin * vkq;
                row[q] = sin * vkp + cos * vkq;
            }
        }
    }
    let mut largest = 0;
    for i in 1..4 {
        if a[i][i] > a[largest][largest] {
            largest = i;
        }
    }
    [
        eigenvectors[0][largest],
        eigenvectors[1][largest],
        eigenvectors[2][largest],
        eigenvectors[3][largest],
    ]
}

/// Convert a (not necessarily normalized) quaternion into a rotation matrix.
fn quaternion_to_rotation(q: [f64; 4]) -> [[f64; 3]; 3] {
    let norm_sq = q.iter().map(|x| x * x).sum::<f64>();
    if norm_sq == 0.0 {
        return IDENTITY;
    }
    let norm = norm_sq.sqrt();
    let [q0, q1, q2, q3] = q.map(|x| x / norm);
    [
        [
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ],
    ]
}
