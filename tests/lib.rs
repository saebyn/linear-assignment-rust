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

    let mut rows: HashSet<usize> = HashSet::new();
    let mut columns: HashSet<usize> = HashSet::new();
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
