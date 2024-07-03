use pyo3::prelude::*;
use num_complex;
use rayon::prelude::*;

type Cplx = num_complex::Complex<f64>;

#[pyclass]
struct Coin {
    is_macro : bool,
    coin : Vec<Vec<Cplx>>,
    coins : Vec<Vec<Vec<Cplx>>>,
}
impl Clone for Coin {
    fn clone(&self) -> Self {
        Coin{is_macro:self.is_macro, coin:self.coin.clone(), coins:self.coins.clone()}
    }
}
impl Coin {
    fn get_coin(&self, e : usize) -> &Vec<Vec<Cplx>> {
        if self.is_macro {
            &self.coin
        } else {
            &self.coins[e]
        }
    }
}
#[pymethods]
impl Coin {
    #[new]
    fn new() -> Self {
        Coin{is_macro:true, coin:Vec::new(), coins:Vec::new()}
    }

    fn set_macro(&mut self, coin : Vec<Vec<Cplx>>) {
        self.is_macro = true;
        self.coin = coin;
        self.coins = Vec::new();
    }

    fn set_micro(&mut self, coins : Vec<Vec<Vec<Cplx>>>) {
        self.is_macro = false;
        self.coin = Vec::new();
        self.coins = coins;
    }
}



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
    fn coin(&mut self, c : &Coin) {
        for i in 0..self.e {
            let op = c.get_coin(i);
            let (u1,u2) = (self.state[2*i],self.state[2*i+1]);
            self.state[2*i] = op[0][0]*u1 + op[0][1]*u2;
            self.state[2*i+1] = op[1][0]*u1 + op[1][1]*u2;
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

    fn oracle(&mut self, search : &Vec<usize>, r : &Vec<Vec<Cplx>>) {
        for i in search.iter() {
            let (u1,u2) = (self.state[2*i],self.state[2*i+1]);
            self.state[2*i] = r[0][0]*u1 + r[0][1]*u2;
            self.state[2*i+1] = r[1][0]*u1 + r[1][1]*u2;
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

    fn run(&mut self, c : Coin, r : Vec<Vec<Cplx>>, ticks : usize, search : Vec<usize>) {
        for _i in 0..ticks {
            self.oracle(&search,&r);
            self.coin(&c);
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

    fn carac(&mut self, c : Coin, r : Vec<Vec<Cplx>>, search : Vec<usize>, waiting : i32) -> PyResult<(usize,f64)> {
        let mut current : f64 = self.get_proba(search.clone()).unwrap();
        let mut min : f64 = current;
        let mut max : f64 = current;
        let mut pos : usize = 0;
        let mut steps : usize = 0;
        let mut waiting = waiting;
        self.reset();

        loop {
            self.run(c.clone(),r.clone(),1,search.clone());
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

}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn qwgraph(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<QWFast>()?;
    m.add_class::<Coin>()?;
    Ok(())
}