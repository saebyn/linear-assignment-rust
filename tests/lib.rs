#![feature(collections)]
// Copyright (c) 2015 John Weaver and contributors.
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate nalgebra as na;
extern crate rand;
extern crate quickcheck;
extern crate linear_assignment;

use std::collections::HashSet;
use std::collections::BitSet;
use std::cmp;
use na::Transpose;
use quickcheck::{TestResult};

use linear_assignment::*;


trait LinearAssignmentProblem {
    fn munkres(&self) -> HashSet<Edge>;
}


impl LinearAssignmentProblem for na::DMat<u32> {
    fn munkres(&self) -> HashSet<Edge> {
        let mut matrix = &mut na::DMat::<u32>::new_zeros(self.nrows(), self.ncols());
        matrix.clone_from(self);
        let transposed = self.nrows() > self.ncols();

        if transposed {
            matrix.transpose_mut();
        }
        let size = MatrixSize { rows: matrix.nrows(), columns: matrix.ncols() };
        assert!(size.columns >= size.rows);
        let edges = solver::<na::DMat<u32>>(&mut matrix, &size);
        if transposed {
            edges.iter().map(|&(v, u)| (u, v)).collect()
        } else {
            edges
        }
    }
}


#[test]
fn solve_1x1() {
    let test_matrix = na::DMat::from_row_vec(
        1,
        1,
        &[0]
    );
    let result = test_matrix.munkres();
    assert!(result.len() == 1);
    assert!(result.contains(&(0,0)));
}

#[test]
fn solve_2x2() {
    let test_matrix = na::DMat::from_row_vec(
        2,
        2,
        &[1, 0,
          0, 1]
    );
    let result = test_matrix.munkres();
    assert!(result.len() == 2);
    assert!(result.contains(&(0,1)));
    assert!(result.contains(&(1,0)));
}

#[test]
fn solve_3x3() {
    let test_matrix = na::DMat::from_row_vec(
        3,
        3,
        &[1, 0, 1,
          0, 1, 1,
          1, 1, 0]
    );
    let result = test_matrix.munkres();
    assert!(result.len() == 3);
    assert!(result.contains(&(0,1)));
    assert!(result.contains(&(1,0)));
    assert!(result.contains(&(2,2)));
}

#[test]
fn solve_3x2_case_1() {
    let test_matrix = na::DMat::from_row_vec(
        3,
        2,
        &[1, 2,
          0, 9,
          9, 9]
    );
    let result = test_matrix.munkres();
    assert!(result.len() == 2);
    println!("{:?}", result);
    assert!(result.contains(&(0,1)));
    assert!(result.contains(&(1,0)));
}

#[test]
fn solve_2x3_case_1() {
    let test_matrix = na::DMat::from_row_vec(
        2,
        3,
        &[1, 0, 9,
          2, 9, 9]
    );
    let result = test_matrix.munkres();
    assert!(result.len() == 2);
    println!("{:?}", result);
    assert!(result.contains(&(0,1)));
    assert!(result.contains(&(1,0)));
}

#[quickcheck]
fn number_of_results_is_k(matrix: na::DMat<u32>) -> TestResult {
    if matrix.ncols() > 10 || matrix.nrows() > 10 {
        return TestResult::discard()
    }

    TestResult::from_bool(
        matrix.munkres().len() == cmp::min(matrix.nrows(), matrix.ncols())
    )
}


#[quickcheck]
fn result_is_an_independent_set(matrix: na::DMat<u32>) -> TestResult {
    if matrix.ncols() > 10 || matrix.nrows() > 10 {
        return TestResult::discard()
    }

    let mut rows = BitSet::new();
    let mut columns = BitSet::new();
    let mut count = 0;

    for (row, column) in matrix.munkres() {
        rows.insert(row);
        columns.insert(column);
        count = count + 1;
    }

    TestResult::from_bool(
        rows.len() == count && columns.len() == count
    )
}


#[quickcheck]
fn identical_results(matrix: na::DMat<u32>) -> TestResult {
    if matrix.ncols() > 10 || matrix.nrows() > 10 {
        return TestResult::discard()
    }

    TestResult::from_bool(
        matrix.munkres() == matrix.munkres()
    )
}
