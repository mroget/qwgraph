use pyo3::prelude::*;
use num_complex;
use rayon::prelude::*;
use indicatif::ParallelProgressIterator;
use indicatif::ProgressIterator;

type Cplx = num_complex::Complex<f64>;



#[pyclass]
struct QWFast {
    #[pyo3(get, set)]
    state: Vec<Cplx>,
    #[pyo3(get, set)]
    wiring: Vec<usize>,
    #[pyo3(get, set)]
    n : usize,
    #[pyo3(get, set)]
    e : usize,
}
impl Clone for QWFast {
    fn clone(&self) -> Self {
        QWFast{state:self.state.clone(),
            wiring:self.wiring.clone(),
            n : self.n,
            e : self.e}
    }
}

impl QWFast {
    fn coin(&mut self, C : &Vec<Vec<Cplx>>) {
        for i in 0..self.e {
            let (u1,u2) = (self.state[2*i],self.state[2*i+1]);
            self.state[2*i] = C[0][0]*u1 + C[0][1]*u2;
            self.state[2*i+1] = C[1][0]*u1 + C[1][1]*u2;
        }
    }

    fn scattering(&mut self) {
        let mut mu : Vec<Cplx> = vec![Cplx::new(0.,0.);self.n];
        let mut size : Vec<usize> = vec![0;self.n];
        for i in 0..(2*self.e) {
            mu[self.wiring[i]] += self.state[i];
            size[self.wiring[i]] += 1;
        }
        for i in 0..mu.len() {
            mu[i] = mu[i]/(size[i] as f64);
        }
        for i in 0..(2*self.e) {
            self.state[i] = 2.*mu[self.wiring[i]] - self.state[i];
        }
    }

    fn oracle(&mut self, search : &Vec<usize>, R : &Vec<Vec<Cplx>>) {
        for i in search.iter() {
            let (u1,u2) = (self.state[2*i],self.state[2*i+1]);
            self.state[2*i] = R[0][0]*u1 + R[0][1]*u2;
            self.state[2*i+1] = R[1][0]*u1 + R[1][1]*u2;
        }
    }
}

#[pymethods]
impl QWFast {
    #[new]
    fn new(wiring : Vec<usize>, n : usize, e : usize) -> Self {
        let mut ret = QWFast {wiring : wiring.clone(), 
                                n : n,
                                e : e,
                                state : Vec::new()};
        ret.reset();
        ret
    }

    fn run(&mut self, C : Vec<Vec<Cplx>>, R : Vec<Vec<Cplx>>, ticks : usize, search : Vec<usize>) {
        for i in 0..ticks {
            self.oracle(&search,&R);
            self.coin(&C);
            self.scattering();
        }
    }

    fn reset(&mut self) {
        self.state = vec![Cplx::new(1./(2.*self.e as f64).sqrt(),0.);2*self.e];
    } 

    fn get_proba(&self, search : Vec<usize>) -> PyResult<f64> {
        let mut p : f64 = 0.;
        for i in search.iter() {
            p+= self.state[2*i].norm().powi(2) + self.state[2*i+1].norm().powi(2);
        }
        Ok(p)
    }

    fn carac(&mut self, C : Vec<Vec<Cplx>>, R : Vec<Vec<Cplx>>, search : Vec<usize>, waiting : i32) -> PyResult<(usize,f64)> {
        let mut current : f64 = self.get_proba(search.clone()).unwrap();
        let mut min : f64 = current;
        let mut max : f64 = current;
        let mut pos : usize = 0;
        let mut steps : usize = 0;
        let mut waiting = waiting;
        self.reset();

        loop {
            self.run(C.clone(),R.clone(),1,search.clone());
            steps+=1;
            current = self.get_proba(search.clone()).unwrap();
            if waiting <= 0 && current < (max+min)/2. {
                break;
            }
            if current > max {
                max = current;
                pos = steps;
            }
            if current < min {
                min = current;
            }
            waiting -= 1;
        }
        Ok((pos,max))
    }

    /*
    fn carac_multiple(&mut self, C : Vec<Vec<Cplx>>, R : Vec<Vec<Cplx>>, args : Vec<(Vec<usize>,PyObject)>, waiting : i32, parallalized : usize, pbar : bool) -> PyResult<Vec<(PyObject,usize,f64)>> {
        let f = |x : &(Vec<usize>,PyObject)| {
            let mut current = self.clone();
            let (search,label) = x;
            let (t,p) = current.carac(C.clone(),R.clone(),search.clone(),waiting).unwrap();
            (label.clone(),t,p)
        };
        if parallalized>1 {
            Ok(args.par_iter().progress_count(args.len() as u64).map(f).collect::<Vec<(PyObject,usize,f64)>>())
        }
        else {
            Ok(args.iter().progress_count(args.len() as u64).map(f).collect::<Vec<(PyObject,usize,f64)>>())
        }
    }*/
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn qwgraph(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<QWFast>()?;
    Ok(())
}