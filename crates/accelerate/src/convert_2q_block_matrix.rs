// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use smallvec::SmallVec;

use faer::modules::core::mul::matmul;
use faer::perm::swap_rows;
use faer::prelude::*;
use faer::{mat, Mat, Parallelism};
use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};

use std::mem::swap;

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, SmallVec<[u8; 2]>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let input_matrix = op_list[0].0.as_array().into_faer_complex();

    let mut matrix = match op_list[0].1.as_slice() {
        [0] => kron_identity_x_matrix_alloc(input_matrix),
        [1] => kron_matrix_x_identity_alloc(input_matrix),
        [0, 1] => input_matrix.to_owned(),
        [1, 0] => change_basis_faer(input_matrix),
        [] => Mat::<c64>::identity(4, 4),
        _ => unreachable!(),
    };

    let mut aux = Mat::<c64>::with_capacity(4, 4);
    let mut aux2 = Mat::<c64>::with_capacity(4, 4);
    // SAFETY: `aux` is a 4x4 matrix whose values are uninitialized and it's used only to store the
    // result of the `matmul` call inside the for loop
    unsafe {
        aux.set_dims(4, 4);
        aux2.set_dims(4, 4);
    };
    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = op_matrix.as_array().into_faer_complex();

        match q_list.as_slice() {
            [0] => {
                kron_identity_x_matrix(&mut aux2, op_matrix);
                matmul_to_dst(&mut aux, aux2.as_ref(), matrix.as_ref());
                swap(&mut aux, &mut matrix);
            }
            [1] => {
                kron_matrix_x_identity(&mut aux2, op_matrix);
                matmul_to_dst(&mut aux, aux2.as_ref(), matrix.as_ref());
                swap(&mut aux, &mut matrix);
            }
            [1, 0] => {
                matmul_to_dst(
                    &mut aux,
                    change_basis_faer(op_matrix).as_ref(),
                    matrix.as_ref(),
                );
                swap(&mut aux, &mut matrix);
            }
            [] => (),
            _ => {
                matmul_to_dst(&mut aux, op_matrix, matrix.as_ref());
                swap(&mut aux, &mut matrix);
            }
        };
    }

    Ok(matrix
        .as_ref()
        .into_ndarray_complex()
        .to_owned()
        .into_pyarray_bound(py)
        .unbind())
}

#[inline]
fn matmul_to_dst(dst: &mut Mat<c64>, lhs: MatRef<c64>, rhs: MatRef<c64>) {
    matmul(
        dst.as_mut(),
        lhs,
        rhs,
        None,
        c64::new(1., 0.),
        Parallelism::None,
    );
}

#[inline]
fn kron_matrix_x_identity_alloc(lhs: MatRef<c64>) -> Mat<c64> {
    let zero: c64 = c64::new(0., 0.);
    mat![
        [lhs[(0, 0)], zero, lhs[(0, 1)], zero],
        [zero, lhs[(0, 0)], zero, lhs[(0, 1)]],
        [lhs[(1, 0)], zero, lhs[(1, 1)], zero],
        [zero, lhs[(1, 0)], zero, lhs[(1, 1)]],
    ]
}

#[inline]
fn kron_identity_x_matrix_alloc(rhs: MatRef<c64>) -> Mat<c64> {
    let zero: c64 = c64::new(0., 0.);
    mat![
        [rhs[(0, 0)], rhs[(0, 1)], zero, zero],
        [rhs[(1, 0)], rhs[(1, 1)], zero, zero],
        [zero, zero, rhs[(0, 0)], rhs[(0, 1)]],
        [zero, zero, rhs[(1, 0)], rhs[(1, 1)]],
    ]
}

#[inline]
fn kron_matrix_x_identity(output: &mut Mat<c64>, oneq_mat: MatRef<c64>) {
    let _zero = c64::new(0., 0.);
    (
        output[(0, 0)],
        output[(0, 1)],
        output[(0, 2)],
        output[(0, 3)],
    ) = (oneq_mat[(0, 0)], _zero, oneq_mat[(0, 1)], _zero);
    (
        output[(1, 0)],
        output[(1, 1)],
        output[(1, 2)],
        output[(1, 3)],
    ) = (_zero, oneq_mat[(0, 0)], _zero, oneq_mat[(0, 1)]);
    (
        output[(2, 0)],
        output[(2, 1)],
        output[(2, 2)],
        output[(2, 3)],
    ) = (oneq_mat[(1, 0)], _zero, oneq_mat[(1, 1)], _zero);
    (
        output[(3, 0)],
        output[(3, 1)],
        output[(3, 2)],
        output[(3, 3)],
    ) = (_zero, oneq_mat[(1, 0)], _zero, oneq_mat[(1, 1)]);
}

#[inline]
fn kron_identity_x_matrix(output: &mut Mat<c64>, oneq_mat: MatRef<c64>) {
    let _zero = c64::new(0., 0.);
    (
        output[(0, 0)],
        output[(0, 1)],
        output[(0, 2)],
        output[(0, 3)],
    ) = (oneq_mat[(0, 0)], oneq_mat[(0, 1)], _zero, _zero);
    (
        output[(1, 0)],
        output[(1, 1)],
        output[(1, 2)],
        output[(1, 3)],
    ) = (oneq_mat[(1, 0)], oneq_mat[(1, 1)], _zero, _zero);
    (
        output[(2, 0)],
        output[(2, 1)],
        output[(2, 2)],
        output[(2, 3)],
    ) = (_zero, _zero, oneq_mat[(0, 0)], oneq_mat[(0, 1)]);
    (
        output[(3, 0)],
        output[(3, 1)],
        output[(3, 2)],
        output[(3, 3)],
    ) = (_zero, _zero, oneq_mat[(1, 0)], oneq_mat[(1, 1)]);
}

/// Switches the order of qubits in a two qubit operation.
/// This function will substitue `change_basis` once the
/// `two_qubit_decompose.rs` uses Mat<c64> instead of ArrayView2
#[inline]
pub fn change_basis_faer(matrix: MatRef<c64>) -> Mat<c64> {
    let mut trans_matrix: Mat<c64> = matrix.transpose().to_owned();
    let (row1, row2) = trans_matrix.as_mut().two_rows_mut(1, 2);
    swap_rows(row1, row2);

    trans_matrix = trans_matrix.transpose().to_owned();
    let (row1, row2) = trans_matrix.as_mut().two_rows_mut(1, 2);
    swap_rows(row1, row2);

    trans_matrix
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}

#[pymodule]
pub fn convert_2q_block_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}
