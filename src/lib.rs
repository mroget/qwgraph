use pyo3::prelude::*;
use num_complex;
use rayon::prelude::*;

#[macro_export]
macro_rules! neighbor {
    ( $x:expr  ) => {
        {
            if $x%2==0 {$x+1} else {$x-1}
        }
    };
}


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


fn get_perm(e : usize, n : usize, wiring : &Vec<usize>) -> Vec<usize> {
    let mut nodes : Vec<Vec<usize>> =  Vec::new();
    for _i in 0..n {nodes.push(Vec::new());}
    for i in 0..(2*e) {
        nodes[wiring[i]].push(i);
    }
    for i in 0..n {
        nodes[i].sort_by(|a, b| wiring[neighbor!(a)].partial_cmp(&wiring[neighbor!(b)]).unwrap());
    }
    let mut perm = vec![0; 2*e];
    for i in 0..n {
        for j in 0..(nodes[i].len()-1) {
            perm[nodes[i][j]] = nodes[i][j+1];
        }
        perm[nodes[i][nodes[i].len()-1]] = nodes[i][0];
    }
    perm
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
    fn apply(&self, e : usize, state : &mut Vec<Cplx>) {
        if self.is_macro {
            for i in 0..e {
                let (u1,u2) = (state[2*i],state[2*i+1]);
                state[2*i] = self.coin[0][0]*u1 + self.coin[0][1]*u2;
                state[2*i+1] = self.coin[1][0]*u1 + self.coin[1][1]*u2;
            }
        }
        else {
            for i in 0..e {
                let (u1,u2) = (state[2*i],state[2*i+1]);
                state[2*i] = self.coins[i][0][0]*u1 + self.coins[i][0][1]*u2;
                state[2*i+1] = self.coins[i][1][0]*u1 + self.coins[i][1][1]*u2;
            }
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
    perm : Option<Vec<usize>>,
}
impl Clone for Scattering {
    fn clone(&self) -> Self {
        Scattering{r#type:self.r#type, fct:self.fct.clone(), perm:self.perm.clone()}
    }
}
impl Scattering {
    fn apply_on_node(&self, u : &Vec<Vec<Cplx>>, targets : &Vec<usize>, state : &mut Vec<Cplx>) {
        assert!(u.len() == u[0].len() && u.len() == targets.len());
        let mut tmp = Vec::with_capacity(targets.len());
        for &i in targets.iter() {
            tmp.push(state[i]);
        }

        tmp = dot(u, &tmp);

        for i in 0..tmp.len() {
            state[targets[i]] = tmp[i];
        }
    }

    fn apply_fct(&self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        let mut nodes : Vec<Vec<usize>> =  Vec::new();
        for _i in 0..n {nodes.push(Vec::new());}
        for i in 0..(2*e) {
            nodes[wiring[i]].push(i);
        }
        for i in 0..n {
            nodes[i].sort_by(|a, b| wiring[neighbor!(a)].partial_cmp(&wiring[neighbor!(b)]).unwrap());
            if self.r#type == 2 {
                self.apply_on_node(&self.fct[nodes[i].len()], &nodes[i], state);
            }
            if self.r#type == 3 {
                self.apply_on_node(&self.fct[i], &nodes[i], state);
            }
        }
    }

    fn apply_grover(&self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        let mut mu : Vec<Cplx> = vec![Cplx::new(0.,0.);n];
        let mut size : Vec<usize> = vec![0;n];
        for i in 0..(2*e) {
            mu[wiring[i]] += state[i];
            size[wiring[i]] += 1;
        }
        for i in 0..mu.len() {
            mu[i] = mu[i]/(size[i] as f64);
        }
        for i in 0..(2*e) {
            state[i] = 2.*mu[wiring[i]] - state[i];
        }
    }

    fn apply_perm(&mut self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        let perm = 
            match &self.perm {
                None => { // create the permutation for the first time
                    self.perm = Some(get_perm(e,n,wiring));
                    self.perm.clone().unwrap()
                },
                Some(x) => {x.clone()},
            };

        // apply the permutation
        let tmp = state.clone();
        for i in 0..(2*e) {
            state[perm[i]] = tmp[i];
        }
    }

    fn apply(&mut self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        match self.r#type {
            0 => {self.apply_perm(e, n, state, wiring);},
            1 => {self.apply_grover(e, n, state, wiring);},
            2 => {self.apply_fct(e, n, state, wiring);},
            3 => {self.apply_fct(e, n, state, wiring);},
            _ => {panic!("Wrong identifier for scattering operator !");},
        }
    }
}
#[pymethods]
impl Scattering {
    #[new]
    fn new() -> Self {
        Scattering{r#type:0, fct:Vec::new(), perm:None}
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
        c.apply(self.e, &mut self.state)
    }

    fn scattering(&mut self, s : &mut Scattering) {
        s.apply(self.e, self.n, &mut self.state, &self.wiring);
    }

    fn oracle(&mut self, search : &Vec<usize>, r : &Vec<Vec<Cplx>>) {
        for i in search.iter() {
            let (u1,u2) = (self.state[2*i],self.state[2*i+1]);
            self.state[2*i] = r[0][0]*u1 + r[0][1]*u2;
            self.state[2*i+1] = r[1][0]*u1 + r[1][1]*u2;
        }
    }

    fn measure(&mut self, search : &Vec<usize>) -> Vec<f64> {
        let mut ret = Vec::new();
        ret.push(0.);

        for i in search.iter() {
            let tmp = self.state[2*i].norm().powi(2) + self.state[2*i+1].norm().powi(2);
            ret[0]+= tmp;
            ret.push(tmp);
        }

        ret
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

    fn get_perm(&self) -> PyResult<Vec<usize>> {
        Ok(get_perm(self.e,self.n,&self.wiring))
    }

    fn run(&mut self, c : Coin, s : &mut Scattering, r : Vec<Vec<Cplx>>, ticks : usize, search : Vec<usize>) {
        for _i in 0..ticks {
            self.oracle(&search,&r);
            self.coin(&c);
            self.scattering(s);
        }
    }

    fn search(&mut self, c : Coin, s : &mut Scattering, r : Vec<Vec<Cplx>>, ticks : usize, search : Vec<usize>) -> PyResult<Vec<Vec<f64>>> {
        let mut ret = Vec::new();
        for _i in 0..ticks {
            ret.push(self.measure(&search));
            self.oracle(&search,&r);
            self.coin(&c);
            self.scattering(s);
        }
        ret.push(self.measure(&search));
        Ok(ret)
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

    fn carac(&mut self, c : Coin, s : &mut Scattering, r : Vec<Vec<Cplx>>, search : Vec<usize>, waiting : i32) -> PyResult<(usize,f64)> {
        let mut current : f64 = self.get_proba(search.clone()).unwrap();
        let mut min : f64 = current;
        let mut max : f64 = current;
        let mut pos : usize = 0;
        let mut steps : usize = 0;
        let mut waiting = waiting;
        self.reset();

        loop {
            self.run(c.clone(),s,r.clone(),1,search.clone());
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