use pyo3::prelude::*;
use num_complex;
use rayon::prelude::*;
use std::f64::consts::PI;

type Cplx = num_complex::Complex<f64>;

fn dot(a : &Vec<Vec<Cplx>>, b : &Vec<Cplx>) -> Vec<Cplx> {
    assert!(a[0].len() == b.len());
    let mut c = b.clone();
    for i in 0..a.len() {
        let mut s = Cplx::new(0.,0.);
        for j in 0..b.len() {
            s += a[i][j] * b[j];
        }   
        c[i] = s;
    }
    c
}


/*************************************************************/
/*******                Coin Class                    ********/
/*************************************************************/

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









/*************************************************************/
/*******              Scattering Class                ********/
/*************************************************************/

#[pyclass]
struct Scattering {
    r#type : usize, // 0: Cycle, 1: Grover, 2: degree fct, 3: node fct
    fct : Vec<Vec<Vec<Cplx>>>, // fct
}
impl Clone for Scattering {
    fn clone(&self) -> Self {
        Scattering{r#type:self.r#type, fct:self.fct.clone()}
    }
}
impl Scattering {
    fn get_op(&self, node : usize, degree : usize) -> Vec<Vec<Cplx>> {
        match self.r#type {
            0 => { // Cycle
                let mut mat = vec![vec![Cplx::new(0.,0.); degree]; degree];
                for i in 0..(degree-1) {
                    mat[i+1][i] = Cplx::new(1.,0.);
                }
                mat[0][degree-1] = Cplx::new(1.,0.);
                mat
            },
            1 => { // Grover
                let mut mat = vec![vec![Cplx::new(0.,0.); degree]; degree];
                for i in 0..degree {
                    for j in 0..degree {
                        mat[i][j] = Cplx::new(2./(degree as f64),0.);
                    }
                    mat[i][i] = mat[i][i] - Cplx::new(1.,0.);
                }
                mat
            },
            2 => { // Degree Function
                self.fct[degree].clone()
            },
            3 => { // Node Function
                self.fct[node].clone()
            },
            _ => {panic!("Wrong identifier for scattering operator !")},
        }
    }
}
#[pymethods]
impl Scattering {
    #[new]
    fn new() -> Self {
        Scattering{r#type:0, fct:Vec::new()}
    }

    fn set_type(&mut self, r#type : usize, fct : Vec<Vec<Vec<Cplx>>>) {
        self.r#type = r#type;
        self.fct = fct;
    }
}















/*************************************************************/
/*******                  QW Class                    ********/
/*************************************************************/


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

    fn apply(&mut self, u : &Vec<Vec<Cplx>>, targets : &Vec<usize>) {
        assert!(u.len() == u[0].len() && u.len() == targets.len());
        let mut tmp = Vec::with_capacity(targets.len());
        for &i in targets.iter() {
            tmp.push(self.state[i]);
        }

        tmp = dot(u, &tmp);

        for i in 0..tmp.len() {
            self.state[targets[i]] = tmp[i];
        }
    }

    fn scattering(&mut self, s : &Scattering) {
        let mut nodes : Vec<Vec<usize>> =  Vec::new();
        for _i in 0..self.n {nodes.push(Vec::new());}
        for i in 0..(2*self.e) {
            nodes[self.wiring[i]].push(i);
        }
        for i in 0..self.n {
            nodes[i].sort();
            self.apply(&s.get_op(i, nodes[i].len()), &nodes[i]);
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

    fn run(&mut self, c : Coin, s : Scattering, r : Vec<Vec<Cplx>>, ticks : usize, search : Vec<usize>) {
        for _i in 0..ticks {
            self.oracle(&search,&r);
            self.coin(&c);
            self.scattering(&s);
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

    fn carac(&mut self, c : Coin, s : Scattering, r : Vec<Vec<Cplx>>, search : Vec<usize>, waiting : i32) -> PyResult<(usize,f64)> {
        let mut current : f64 = self.get_proba(search.clone()).unwrap();
        let mut min : f64 = current;
        let mut max : f64 = current;
        let mut pos : usize = 0;
        let mut steps : usize = 0;
        let mut waiting = waiting;
        self.reset();

        loop {
            self.run(c.clone(),s.clone(),r.clone(),1,search.clone());
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
    m.add_class::<Scattering>()?;
    Ok(())
}