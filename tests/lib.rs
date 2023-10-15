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

extern crate nalgebra as na;

use std::collections::HashSet;

use linear_assignment::*;

trait LinearAssignmentProblem {
    fn munkres(&self) -> HashSet<Edge>;
}

impl LinearAssignmentProblem for na::DMatrix<u32> {
    fn munkres(&self) -> HashSet<Edge> {
        let transposed = self.nrows() > self.ncols();
        let mut matrix = {
            if transposed {
                self.transpose()
            } else {
                self.clone()
            }
        };
        let size = MatrixSize {
            rows: matrix.nrows(),
            columns: matrix.ncols(),
        };
        assert!(size.columns >= size.rows);
        let edges = solver::<na::DMatrix<u32>>(&mut matrix, &size);
        if transposed {
            edges.iter().map(|&(v, u)| (u, v)).collect()
        } else {
            edges
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use bit_set::BitSet;
    use quickcheck::{quickcheck, TestResult};

    use crate::LinearAssignmentProblem;

    #[test]
    fn solve_1x1() {
        let test_matrix = na::DMatrix::from_row_slice(1, 1, &[0]);
        let result = test_matrix.munkres();
        assert!(result.len() == 1);
        assert!(result.contains(&(0, 0)));
    }
    #[test]
    fn solve_2x2_case_1() {
        let test_matrix = na::DMatrix::from_row_slice(2, 2, &[1, 0, 0, 1]);
        let result = test_matrix.munkres();
        assert!(result.len() == 2);
        assert!(result.contains(&(0, 1)));
        assert!(result.contains(&(1, 0)));
    }
    #[test]
    fn solve_2x2_case_2() {
        let test_matrix = na::DMatrix::from_row_slice(2, 2, &[0, 1, 1, 0]);
        let result = test_matrix.munkres();
        assert!(result.len() == 2);
        assert!(result.contains(&(0, 0)));
        assert!(result.contains(&(1, 1)));
    }
    #[test]
    fn solve_3x3_case_1() {
        let test_matrix = na::DMatrix::from_row_slice(3, 3, &[1, 0, 1, 0, 1, 1, 1, 1, 0]);
        let result = test_matrix.munkres();
        assert!(result.len() == 3);
        assert!(result.contains(&(0, 1)));
        assert!(result.contains(&(1, 0)));
        assert!(result.contains(&(2, 2)));
    }
    #[test]
    fn solve_3x2_case_1() {
        let test_matrix = na::DMatrix::from_row_slice(3, 2, &[1, 2, 0, 9, 9, 9]);
        let result = test_matrix.munkres();
        assert!(result.len() == 2);
        println!("{:?}", result);
        assert!(result.contains(&(0, 1)));
        assert!(result.contains(&(1, 0)));
    }
    #[test]
    fn solve_2x3_case_1() {
        let test_matrix = na::DMatrix::from_row_slice(2, 3, &[1, 0, 9, 2, 9, 9]);
        let result = test_matrix.munkres();
        assert!(result.len() == 2);
        println!("{:?}", result);
        assert!(result.contains(&(0, 1)));
        assert!(result.contains(&(1, 0)));
    }
    #[test]
    fn solve_3x3_case_2() {
        let test_matrix = na::DMatrix::from_row_slice(3, 3, &[1, 0, 9, 2, 9, 9, 10, 10, 10]);
        let result = test_matrix.munkres();
        assert!(result.len() == 3);
        println!("{:?}", result);
        assert!(result.contains(&(0, 1)));
        assert!(result.contains(&(1, 0)));
        assert!(result.contains(&(2, 2)));
    }
    #[test]
    fn solve_issue_6() {
        let test_matrix = na::DMatrix::from_row_slice(
            5,
            5,
            &[
                3, 6, 2, 5, 7, 2, 3, 1, 4, 3, 8, 1, 4, 2, 2, 6, 3, 1, 4, 4, 3, 2, 5, 2, 2,
            ],
        );
        let result = test_matrix.munkres();
        println!("{:?}", result);
        assert_eq!(result.len(), 5);
        let cost = result.iter().fold(0, |acc, edge| acc + test_matrix[*edge]);
        assert_eq!(cost, 10);
    }
    #[test]
    fn solve_issue_10() {
        let test_matrix = na::DMatrix::from_row_slice(
            3, 4,

            &[ 8, 5, 9, 9,
               4, 2, 6, 4,
               7, 3, 7, 8, ]
        );

        let result = test_matrix.munkres();
        println!("{:?}", result);
        assert_eq!(result.len(), 3);
        let cost = result.iter().fold(0, |acc, edge| acc + test_matrix[*edge]);
        assert_eq!(cost, 15);
    }

    quickcheck! {
        fn number_of_results_is_k(matrix: na::DMatrix<u32>) -> TestResult {
            if matrix.ncols() > 10 || matrix.nrows() > 10 {
                return TestResult::discard();
            }

            TestResult::from_bool(matrix.munkres().len() == cmp::min(matrix.nrows(), matrix.ncols()))
        }


        fn result_is_an_independent_set(matrix: na::DMatrix<u32>) -> TestResult {
            if matrix.ncols() > 10 || matrix.nrows() > 10 {
                return TestResult::discard();
            }

            let mut rows = BitSet::new();
            let mut columns = BitSet::new();
            let mut count = 0;

            for (row, column) in matrix.munkres() {
                rows.insert(row);
                columns.insert(column);
                count = count + 1;
            }

            TestResult::from_bool(rows.len() == count && columns.len() == count)
        }


        fn identical_results(matrix: na::DMatrix<u32>) -> TestResult {
            if matrix.ncols() > 10 || matrix.nrows() > 10 {
                return TestResult::discard();
            }

            TestResult::from_bool(matrix.munkres() == matrix.munkres())
        }
    }
}
