#![feature(collections)]
//! This is a library for solving the [linear assignment
//! problem](http://en.wikipedia.org/wiki/Assignment_problem).
//! For this problem, it is helpful to think of the data as a [bipartite
//! graph](http://en.wikipedia.org/wiki/Bipartite_graph),
//! with weighted, undirected edges fully connecting all elements of set `U`
//! to all elements of set `V`.
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
use std::fmt::Debug;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Add;
use std::ops::Sub;
use std::cmp;
use std::collections::HashSet;
use std::collections::BitSet;
use std::iter::FromIterator;

extern crate num;

use num::Zero;

#[macro_use]
extern crate log;


/// An edge between `U` and `V`: (u, v)
pub type Edge = (usize, usize);

pub trait Weight: Zero + Add<Output=Self> + Sub<Output=Self> + Ord + Copy + Debug {}
impl<T: Zero + Add<Output=T> + Sub<Output=T> + Ord + Copy + Debug> Weight for T {}



pub struct MatrixSize {
    pub rows: usize,
    pub columns: usize,
}


/// Find a solution to the assignment problem in `matrix`.
///
/// Given a bipartite graph consisting of the sets `U` and `V`,
/// `matrix` represents the weights of the edges `E` between each vertex
/// represented by the elements of the set `U` and the set `V`. 
///
/// `matrix` should be a rectangular matrix, where `size.columns` >= `size.rows`.
///
/// Here we are interested in finding some subset of the edges `E` of our bipartite graph,
/// (represented by our `matrix`), where:
///
///   * Every vertex in `U` and in `V` is connected via exactly a single edge (not more or less)
///
///   * The sum of the weights of these edges are at least as small as any other subset that
///   satisfies the first condition.
///
/// These two properties must hold for our result.
///
/// Warning: there is the potential for overflow if the values in `matrix` are larger than 1/2 the
/// maximum representable value of type `T::Output`.
///
pub fn solver<T>(matrix: &mut T, size: &MatrixSize) -> HashSet<Edge>
    where T: IndexMut<Edge>,
          T::Output: Weight {
    debug_assert!(size.columns >= size.rows);
    let k = cmp::min(size.columns, size.rows);
    // continual invariant: all values in `matrix` must be >= 0.

    if size.columns == 0 || size.rows == 0 {
        return HashSet::new();
    }

    // For set up, we need to change `matrix` so that every column and every row has at least one
    // zero. because we are only interested in returning the edges from `U` to `V` that represent
    // the smallest sum of weights, and not the sum of weights itself, we don't need to retain the
    // original values. 
    reduce_edges(matrix, size);

    // the algorithm proceeds by "starring" zero weight edges that are optimal with respect to the
    // graph containing only edges that we have starred.
    let mut stars: HashSet<Edge> = HashSet::new();

    // additional set up: star every zero weight edge, such that no starred edges should be adjacent.
    star_isolated_set_of_zeros(&mut stars, matrix, size);

    // the algorithm also "primes" zeros that are candidates for starring in its next iteration.
    let mut primes: HashSet<Edge> = HashSet::new();

    // We "cover" rows and column to exclude them from consideration when looking for zeros to
    // prime.
    let mut columns_covered = BitSet::new();
    let mut rows_covered = BitSet::new();

    let mut covered_column_count = 0;

    // Cover all of the starred columns, then
    // If all columns are now covered, we have a solution.
    loop {
        cover_starred_columns(&mut columns_covered, &stars);

        // assert that we make progress here.
        debug_assert!(columns_covered.len() > covered_column_count);
        covered_column_count = columns_covered.len();

        debug_assert!(columns_covered.len() <= k);
        if columns_covered.len() == k {
            break;
        }

        // Otherwise, we proceed with the algorithm.

        while prime_zeros(&*matrix, &size, &mut columns_covered, &mut rows_covered, &mut stars, &mut primes) {
            debug_assert!(
                all_stars_covered(&stars, &size, &columns_covered, &rows_covered),
                "stars = {:?}, columns covered = {:?}, rows covered = {:?}",
                stars, columns_covered, rows_covered
            );
            debug_assert!(
                stars.intersection(&primes).cloned().collect::<HashSet<_>>().len() == 0,
                "The set of stars and primes intersect! stars = {:?}, primes = {:?}.",
                stars,
                primes
            );
            debug_assert!(
                find_uncovered_zero(&*matrix, &size, &columns_covered, &rows_covered) == None
            );

            adjust_weights(matrix, &size, &mut columns_covered, &mut rows_covered);

        }
    }
    // Once here, we have found our solution in the `stars`.

    // there should be exactly `size.rows` edges in the returned result.
    debug_assert!(stars.len() == size.rows);
    stars
}


/// Prime all uncovered zeros, covering its row, and uncovering its column.
/// If we prime a zero, and it's row has no starred zero, we want to find the "alternating path"
/// of primed and starred zeros in `matrix`, update everything, and then quit.
fn prime_zeros<T>(matrix: &T, size: &MatrixSize, 
                  columns_covered: &mut BitSet, rows_covered: &mut BitSet, 
                  stars: &mut HashSet<Edge>, primes: &mut HashSet<Edge>) -> bool
    where T: Index<Edge>,
          T::Output: Weight {
    // Things to think about:
    //   - the priming procedure doesn't change the weights, so we can cache knowledge about
    //     where the zeros are
    //   - we scan each uncovered row (there may be covered rows that we didn't cover in
    //     this pass, from a previous pass that was interrupted by "alternate path")
    //   - we don't ever uncover a row here
    // TODO collect the position of all zeros in `matrix`, since this won't change until
    // this loop restarts.
    // TODO could we find a better data structure for this?
    // TODO how does doing this here affect complexity?
    // TODO collect which rows have stars, since this won't change until the loop restarts.
    // TODO could we find a better data structure for this?
    // TODO how does doing this here affect complexity?
    loop {
        debug_assert!(
            all_stars_covered(stars, size, columns_covered, rows_covered),
            "stars = {:?}, columns covered = {:?}, rows covered = {:?}",
            stars, columns_covered, rows_covered
        );

        match find_uncovered_zero(matrix, size, columns_covered, rows_covered) {
            Some(edge_to_prime) => {
                debug_assert!(!primes.contains(&edge_to_prime));
                debug_assert!(!stars.contains(&edge_to_prime));
                // prime this zero edge
                primes.insert(edge_to_prime);
                // if there's a starred zero in this row,
                if stars.iter().any(|&(row, _)| row == edge_to_prime.0) {
                    // cover this row, uncover this column
                    rows_covered.insert(edge_to_prime.0);
                    debug_assert!(
                        rows_covered.len() < size.rows,
                        "prime_zeros covered the last row! row = {:?}, stars = {:?}, rows_covered = {:?}, columns_covered = {:?}",
                        edge_to_prime.0,
                        stars,
                        rows_covered,
                        columns_covered
                    );
                    columns_covered.remove(&edge_to_prime.1);
                } else {
                    let path = find_alternating_path(edge_to_prime, &*stars, &*primes);
                    *stars = get_stars_from_path(path, stars);
                    columns_covered.clear();
                    rows_covered.clear();
                    primes.clear();

                    return false;
                }
            },
            None => return true,
        }
    }
}


fn adjust_weights<T>(matrix: &mut T, size: &MatrixSize, columns_covered: &mut BitSet, rows_covered: &mut BitSet)
    where T: IndexMut<Edge>,
          T::Output: Weight {
    // we want to know now, what the smallest uncovered value is.
    // TODO we might be able to find this while in prime_zeros()
    let smallest = find_smallest_uncovered(&*matrix, &size, &columns_covered, &rows_covered);

    // adjust weights
    for row in 0..size.rows {
        for column in 0..size.columns {
            if rows_covered.contains(&row) {
                matrix[(row, column)] = matrix[(row, column)] + smallest;
            }
            if !columns_covered.contains(&column) {
                matrix[(row, column)] = matrix[(row, column)] - smallest;
            }
            debug_assert!(matrix[(row, column)] >= T::Output::zero());
        }
    }
}


fn all_stars_covered(stars: &HashSet<Edge>,  size: &MatrixSize,columns_covered: &BitSet, rows_covered: &BitSet) -> bool {
    for row in 0..size.rows {
        if !rows_covered.contains(&row) {
            for column in 0..size.columns {
                if !columns_covered.contains(&column) {
                    if stars.contains(&(row, column)) {
                        return false;
                    }
                }
            }
        }
    }
    true
}


fn find_uncovered_zero<T>(matrix: &T, size: &MatrixSize, 
                          columns_covered: &BitSet, rows_covered: &BitSet) -> Option<Edge> 
    where T: Index<Edge>,
          T::Output: Weight {
    for row in 0..size.rows {
        if !rows_covered.contains(&row) {
            for column in 0..size.columns {
                if !columns_covered.contains(&column) {
                    if matrix[(row, column)] == T::Output::zero() {
                        return Some((row, column));
                    }
                }
            }
        }
    }

    None::<Edge>
}


fn find_smallest_uncovered<T>(matrix: &T, size: &MatrixSize, 
                              columns_covered: &BitSet, rows_covered: &BitSet) -> T::Output
    where T: Index<Edge>,
          T::Output: Weight {
    debug_assert!(size.rows > 0 && size.columns > 0);
    debug_assert!(columns_covered.len() < size.columns);
    debug_assert!(rows_covered.len() < size.rows);
    let mut smallest = None;

    for row in 0..size.rows {
        if !rows_covered.contains(&row) {
            for column in 0..size.columns {
                if !columns_covered.contains(&column) {
                    smallest = match smallest {
                        Some(smaller) => Some(cmp::min(smaller, matrix[(row, column)])),
                        None => Some(matrix[(row, column)]),
                    };
                }
            }
        }
    }

    match smallest {
        Some(value) => {
            debug_assert!(value > T::Output::zero());
            value
        },
        None => panic!(),
    }
}



fn find_alternating_path(starting_edge: Edge,
                         stars: &HashSet<Edge>, primes: &HashSet<Edge>) -> Vec<Edge> {
    let mut path = vec![starting_edge];

    debug_assert!(
        stars.intersection(&primes).cloned().collect::<HashSet<_>>().len() == 0,
        "The set of stars and primes intersect! stars = {:?}, primes = {:?}.",
        stars,
        primes
    );

    loop {
        let used: HashSet<Edge> = HashSet::from_iter(path.clone());
        let primes: HashSet<Edge> = primes.difference(&used).map(|&x| x).collect();
        let stars: HashSet<Edge> = stars.difference(&used).map(|&x| x).collect();

        // z0 is the last found primed zero.
        let z0: Edge = match path.last() {
            Some(&z0) => z0,
            None => panic!(),
        };


        match stars.iter().find(|&&(_, column)| column == z0.1) {
            // z1 is (if it exists) the starred zero in the same column as z0.
            Some(&z1) => {
                debug_assert!(!path.contains(&z1));
                path.push(z1);
                // If z1 exists, then there must be a primed zero in the same row as it,
                // and we'll call this primed zero z2.
                // In the next iteration, z0 will take on the value of z2 from here.
                match primes.iter().find(|&&(row, _)| row == z1.0) {
                    Some(&z2) => {
                        debug_assert!(z0 != z2);
                        debug_assert!(!path.contains(&z2), "path = {:?}, z2 = {:?}", path, z2);
                        path.push(z2);
                    },
                    None => panic!(),
                };
            },
            None => break,
        }
    }

    path
}


fn get_stars_from_path(path: Vec<Edge>, stars: &mut HashSet<Edge>) -> HashSet<Edge> {
    let path = path.into_iter().enumerate();
    let mut new_stars: HashSet<Edge> = HashSet::new();
    let mut old_stars: HashSet<Edge> = HashSet::new();

    for (i, edge) in path {
        if i % 2 == 0 {
            new_stars.insert(edge);
        } else {
            old_stars.insert(edge);
        }
    }

    let stars: HashSet<Edge> = stars.difference(&old_stars).map(|&edge| edge).collect();
    stars.union(&new_stars).cloned().collect()
}


fn subtract_from_matrix<T, F>(matrix: &mut T, size: &MatrixSize, f: F)
    where F: Fn(usize, usize) -> T::Output,
          T: IndexMut<Edge>,
          T::Output: Weight {
    for row in 0..size.rows {
        for column in 0..size.columns {
            matrix[(row, column)] = matrix[(row, column)] - f(row, column);
        }
    }
}

fn find_smallest_vector<T, F>(matrix: &T, outer_size: usize, inner_size: usize, f: F) -> Vec<T::Output> 
    where F: Fn(usize, usize) -> Edge,
          T: IndexMut<Edge>,
          T::Output: Weight {
    let mut smallest_in_outside = Vec::new();
    for outer in 0..outer_size {
        // Take the first as initial smallest values, then we'll search the remaining
        // to find the smallest value for each.
        smallest_in_outside.push(matrix[f(outer, 0)]);
        for inner in 1..inner_size {
            let weight = matrix[f(outer, inner)];
            if weight < smallest_in_outside[outer] {
                smallest_in_outside[outer] = weight;
            }
        }
    }

    smallest_in_outside
}


/// We perform a reduction step over each row, then each column, subtracting
/// the smallest value of each from every element in that row or column.
/// this step will ensure that every row and every column has at least one zeroed element. 
fn reduce_edges<'a, T>(matrix: &'a mut T, size: &MatrixSize) -> &'a mut T
    where T: IndexMut<Edge>,
          T::Output: Weight {
    let smallest_in_column = find_smallest_vector(matrix, size.columns, size.rows, |column, row| (row, column));
    // subtract the smallest value of a column from each value in that column
    subtract_from_matrix(matrix, size, |_, column| smallest_in_column[column]);

    let smallest_in_row = find_smallest_vector(matrix, size.rows, size.columns, |row, column| (row, column));
    // ... and then do the same for each row.
    subtract_from_matrix(matrix, size, |row, _| smallest_in_row[row]);

    // assertion: every row and column has at least one zero.
    matrix
}


fn star_isolated_set_of_zeros<'a, T>(stars: &'a mut HashSet<Edge>, matrix: &T, size: &MatrixSize) -> &'a mut HashSet<Edge>
    where T: IndexMut<Edge>,
          T::Output: Weight {
    let mut columns = BitSet::new();

    for row in 0..size.rows {
        for column in 0..size.columns {
            // now that we've found a zero in the this column, we want to not ever consider any
            // zeros in this same column.
            if columns.contains(&column) {
                break;
            }

            if matrix[(row, column)] == T::Output::zero() {
                columns.insert(column);
                stars.insert((row, column));
                break;
            }
        }
        // because we only visit a row once, if we find a zero on a row, we don't have to worry
        // about finding another one, since we stop at the column where we found it and continue
        // onto the next row.
    }

    stars
}


fn cover_starred_columns(cover: &mut BitSet, stars: &HashSet<Edge>) {
    cover.clear();
    cover.extend(stars.iter().map(|x| x.1));
}
