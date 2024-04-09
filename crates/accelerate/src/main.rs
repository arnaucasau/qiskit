use ndarray::Array;
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
    let a: Array<Complex64,_> = Array::from_elem((size, size), Complex64::new(rand_num(), rand_num()));
    let b: Array<Complex64,_> = Array::from_elem((size, size), Complex64::new(rand_num(), rand_num()));
    let mut dst: Array<Complex64,_> = Array::zeros([size, size]);

    let start = Instant::now();
    for _ in 0..N {
        dst = ALPHA * dst + BETA * a.dot(&b);

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

/*
After 100 executions:

4x4 matrices:
    ndarray:
        min: 402ns
        avg: 467.4ns
        max: 781ns

    faer:
        min: 78ns
        avg: 93.83ns
        max: 288ns

All times at:
    ndarray-4.txt: 4x4 matrix 
    faer-4.txt: 4x4 matrix
    
	ndarray-8.txt: 8x8 matrix
    faer-8.txt: 8x8 matrix

	ndarray-16.txt: 16x16 matrix
    faer-16.txt: 16x16 matrix
*/