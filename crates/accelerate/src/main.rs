use ndarray::Array;
use ndarray::prelude::aview2;
use std::time::Instant;
use faer::{Mat, Parallelism};
use faer::modules::core::mul::matmul;
use num_complex::Complex64;

use rand::distributions::{Distribution, Uniform};

use std::env;

const ALPHA: Complex64 = Complex64::new(1.,1.);
const BETA: Complex64 = Complex64::new(0.5,3.);
const N: u32 = 1000;

fn rand_num()-> f64{
    let step = Uniform::new(-1., 1.);
    let mut rng = rand::thread_rng();
    let choice = step.sample(&mut rng);
    return choice;
}

fn ndarray_mat_mul(size: usize){
    // ndarray using Array
    let a: Array<Complex64,_> = Array::from_elem((size, size), Complex64::new(rand_num(), rand_num()));
    let b: Array<Complex64,_> = Array::from_elem((size, size), Complex64::new(rand_num(), rand_num()));

    // ndarray using aview2 fixed to 4x4
    let data_a_aview: [[Complex64; 4]; 4] = [[Complex64::new(rand_num(), rand_num());4];4];
    let a_aview = aview2(&data_a_aview);
    let data_b_aview: [[Complex64; 4]; 4] = [[Complex64::new(rand_num(), rand_num());4];4];
    let b_aview = aview2(&data_b_aview);

    let mut dst: Array<Complex64,_> = Array::zeros([size, size]);

    let start = Instant::now();
    for _ in 0..N {
        // Equivalence of the faer's matmul using Array
        //dst = ALPHA * dst + BETA * a.dot(&b);

        // Equivalence of the faer's matmul using Array
        dst = ALPHA * dst + BETA * a_aview.dot(&b_aview);

        // Simple multiplication
        //dst = a.dot(&b);
    }
    println!("Elapsed time ndarray: {:.2?}", start.elapsed()/N);
}

fn faer_mat_mul(size: usize){
    let a = Mat::<Complex64>::from_fn(size, size, |_, _| Complex64::new(rand_num(), rand_num()));
    let b = Mat::<Complex64>::from_fn(size, size, |_, _| Complex64::new(rand_num(), rand_num()));
    let mut dst = Mat::<Complex64>::zeros(size, size);
    
    let start = Instant::now();
    for _ in 0..N {
        // matmul -> dst = ALPHA * dst + BETA * &a * &b;
        matmul(dst.as_mut(), a.as_ref(), b.as_ref(), Some(ALPHA), BETA, Parallelism::None);

        // Simple multiplication
        //dst = &a * &b;
    }
    println!("Elapsed time faer: {:.2?}", start.elapsed()/N);
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

    let size = match arg.parse::<usize>() {
        Ok(val) => val,
        Err(e) => {
            println!("Unable to parse number from argument: {}", e);
            return;
        }
    };

    ndarray_mat_mul(size);
    faer_mat_mul(size);
}
