use faer::modules::core::mul::matmul;
use faer::{Mat, Parallelism};
use ndarray::prelude::aview2;
use ndarray::Array;
use num_complex::Complex64;
use rand::distributions::{Distribution, Uniform};
use std::time::Instant;
use std::mem::swap;
use std::env;

const ALPHA: Complex64 = Complex64::new(1., 0.);

const ITERATIONS: u32 = 5;
const N: usize = 4; // size of the matrices

fn rand_num() -> f64 {
    let step = Uniform::new(-1., 1.);
    let mut rng = rand::thread_rng();
    let choice = step.sample(&mut rng);
    return choice;
}

fn ndarray_mat_mul() {
    let data_a_aview: [[Complex64; N]; N] = [[Complex64::new(rand_num(), rand_num()); N]; N];
    let a_aview = aview2(&data_a_aview);

    let mut dst: Array<Complex64, _> = Array::from_shape_fn((N, N), |_| Complex64::new(rand_num(), rand_num()));

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        dst = a_aview.dot(&dst);
    }

    println!("{:.2?}", start.elapsed());
}

fn faer_mul() {
    let a = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));
    let mut dst = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        dst = a.as_ref() * dst.as_ref();
    }

    println!("{:.2?}", start.elapsed());
}
fn faer_mat_mul_clone() {
    let a = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));
    let mut dst = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let aux = dst.clone();
        matmul(
            dst.as_mut(),
            a.as_ref(),
            aux.as_ref(),
            None,
            ALPHA,
            Parallelism::None,
        );
    }

    println!("{:.2?}", start.elapsed());
}
fn faer_mat_mul_swap() {
    let a = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));
    let mut dst = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));

    let start = Instant::now();
    let mut aux = Mat::zeros(4, 4);
    for _ in 0..ITERATIONS {
        matmul(
            aux.as_mut(),
            a.as_ref(),
            dst.as_ref(),
            None,
            ALPHA,
            Parallelism::None,
        );
        swap(&mut aux, &mut dst);
    }

    println!("{:.2?}", start.elapsed());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let arg = match args.get(1) {
        Some(val) => val,
        None => {
            println!("Not enough arguments provided!");
            return;
        }
    };

    let method = match arg.parse::<i32>() {
        Ok(val) => val,
        Err(e) => {
            println!("Unable to parse number from argument: {}", e);
            return;
        }
    };

    if method == 1 || method == 5 {
        ndarray_mat_mul();
    }

    if method == 2 || method == 5 {
        faer_mul();
    }

    if method == 3 || method == 5 {
        faer_mat_mul_clone();
    }

    if method == 4 || method == 5 {
        faer_mat_mul_swap();
    }
}
