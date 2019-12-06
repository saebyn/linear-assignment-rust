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
extern crate criterion;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate rand;
use rand::Rng;

extern crate linear_assignment;
extern crate nalgebra as na;

use std::collections::HashSet;

use linear_assignment::*;

trait LinearAssignmentProblem {
    fn munkres(&self) -> HashSet<Edge>;
}

impl LinearAssignmentProblem for na::DMatrix<u32> {
    fn munkres(&self) -> HashSet<Edge> {
        let mut matrix = &mut na::DMatrix::<u32>::zeros(self.nrows(), self.ncols());
        matrix.clone_from(self);
        let transposed = self.nrows() > self.ncols();

        if transposed {
            matrix.transpose_mut();
        }
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

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    c.bench_function("square matrix", |b| {
        b.iter(|| {
            let matrix = black_box(na::DMatrix::<u32>::from_fn(100, 100, |_x, _y| rng.gen()));
            matrix.munkres()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
