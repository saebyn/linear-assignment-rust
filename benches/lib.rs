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

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    c.bench_function("square matrix", |b| {
        b.iter(|| {
            let matrix = black_box(na::DMatrix::<u32>::from_fn(100, 100, |_x, _y| rng.gen()));
            linear_assignment::solver(&mut matrix.clone())
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
