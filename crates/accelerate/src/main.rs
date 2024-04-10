use ndarray::Array;
use ndarray::prelude::aview2;
use std::time::Instant;
use faer::{Mat, Parallelism};
use faer::modules::core::mul::matmul;
use num_complex::Complex64;
use rand::distributions::{Distribution, Uniform};

use std::env;

const ALPHA: Complex64 = Complex64::new(1.,0.);
const BETA: Complex64 = Complex64::new(0.5,0.5);

const ITERATIONS: u32 = 1000;
const N: usize = 4; // size of the matrices


fn rand_num()-> f64{
    let step = Uniform::new(-1., 1.);
    let mut rng = rand::thread_rng();
    let choice = step.sample(&mut rng);
    return choice;
}

fn ndarray_mat_mul(){
    let data_a_aview: [[Complex64; N]; N] = [[Complex64::new(rand_num(), rand_num());N];N];
    let a_aview = aview2(&data_a_aview);
    let data_b_aview: [[Complex64; N]; N] = [[Complex64::new(rand_num(), rand_num());N];N];
    let b_aview = aview2(&data_b_aview);

    let mut dst: Array<Complex64,_> = Array::zeros([N, N]);

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        // Equivalence of the faer's matmul using dot
        dst = ALPHA * dst + BETA * a_aview.dot(&b_aview);
    }

    //println!("Elapsed time ndarray: {:.2?}", start.elapsed()/ITERATIONS);
    println!("{:.2?}", start.elapsed()/ITERATIONS);
}

fn faer_mat_mul(){
    let a = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));
    let b = Mat::<Complex64>::from_fn(N, N, |_, _| Complex64::new(rand_num(), rand_num()));
    let mut dst = Mat::<Complex64>::zeros(N, N);
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        // matmul -> dst = ALPHA * dst + BETA * &a * &b;
        matmul(dst.as_mut(), a.as_ref(), b.as_ref(), Some(ALPHA), BETA, Parallelism::None);
    }

    //println!("Elapsed time faer: {:.2?}", start.elapsed()/ITERATIONS);
    println!("{:.2?}", start.elapsed()/ITERATIONS);
}

fn main(){
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

    if method == 1 || method == 3 {
        ndarray_mat_mul();
    }
    
    if method == 2 || method == 3 {
        faer_mat_mul();
    }
}
