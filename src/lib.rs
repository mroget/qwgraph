use std::io::Write;
use std::io;
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

fn get_indices_around_nodes(e : usize, n : usize, wiring : &Vec<usize>) -> Vec<Vec<usize>> {
    let mut nodes : Vec<Vec<usize>> =  Vec::new();
    for _i in 0..n {nodes.push(Vec::new());}
    for i in 0..(2*e) {
        nodes[wiring[i]].push(i);
    }
    for i in 0..n {
        nodes[i].sort_by(|a, b| wiring[neighbor!(a)].partial_cmp(&wiring[neighbor!(b)]).unwrap());
    }
    nodes
}

#[pyfunction]
fn _get_indices_around_nodes(e : usize, n : usize, wiring : Vec<usize>) -> PyResult<Vec<Vec<usize>>> {
    Ok(get_indices_around_nodes(e, n, &wiring))
}


fn get_perm(e : usize, n : usize, wiring : &Vec<usize>) -> Vec<usize> {
    let nodes = get_indices_around_nodes(e, n, wiring);
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
/*******                Unitary Class                 ********/
/*************************************************************/

#[pyclass]
#[derive(Clone)]
struct UnitaryOp {
    target : Vec<usize>,
    unitary : Vec<Vec<Cplx>>,
}
impl UnitaryOp {
    fn apply(&self, state : &mut Vec<Cplx>) {
        let mut tmp = Vec::with_capacity(self.target.len());
        for &i in self.target.iter() {
            tmp.push(state[i]);
        }

        tmp = dot(&self.unitary, &tmp);

        for i in 0..tmp.len() {
            state[self.target[i]] = tmp[i];
        }
    }
}

#[pymethods]
impl UnitaryOp {
    #[new]
    fn new(target : Vec<usize>, unitary : Vec<Vec<Cplx>>) -> Self {
        UnitaryOp{target : target, unitary : unitary}
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
        let nodes = get_indices_around_nodes(e, n, wiring);
        for i in 0..n {
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

    fn apply_perm(&self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        let perm = get_perm(e,n,wiring);

        // apply the permutation
        let tmp = state.clone();
        for i in 0..(2*e) {
            state[perm[i]] = tmp[i];
        }
    }

    fn apply(&self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
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
        Scattering{r#type:0, fct:Vec::new()}
    }

    fn set_type(&mut self, r#type : usize, fct : Vec<Vec<Vec<Cplx>>>) {
        self.r#type = r#type;
        self.fct = fct;
    }
}






/*************************************************************/
/*******              Operation Class                 ********/
/*************************************************************/
#[derive(Clone)]
enum Operation {
    Scattering(Scattering),
    Coin(Coin),
    Apply(UnitaryOp),
    Proba(Vec<usize>),
    Nothing,
}
impl Operation {
    fn apply(&self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        match self {
            Operation::Scattering(s) => {s.apply(e, n, state, wiring)},
            Operation::Coin(c) => {c.apply(e,state)},
            Operation::Apply(u) => {u.apply(state)},
            Operation::Proba(_) => {},
            Operation::Nothing => {},
        }
    }
}


#[pyclass]
#[derive(Clone)]
struct OperationWrapper {
    op : Operation,
}
impl OperationWrapper {
    fn apply(&self, e : usize, n : usize, state : &mut Vec<Cplx>, wiring : &Vec<usize>) {
        self.op.apply(e, n, state, wiring);
    }
}
#[pymethods]
impl OperationWrapper {
    #[new]
    fn new() -> Self {
        OperationWrapper{op : Operation::Nothing}
    }

    fn set_to_coin(&mut self, c : Coin) {
        self.op = Operation::Coin(c);
    }
    fn set_to_scattering(&mut self, s : Scattering) {
        self.op = Operation::Scattering(s);
    }
    fn set_to_unitary(&mut self, u : UnitaryOp) {
        self.op = Operation::Apply(u);
    }
    fn set_to_proba(&mut self, targets : Vec<usize>) {
        self.op = Operation::Proba(targets);
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
    fn proba(&self, target : &Vec<usize>) -> f64 {
        let mut p : f64 = 0.;
        for &i in target.iter() {
            p+= self.state[i].norm().powi(2);
        }
        p
    }

    fn apply(&mut self, pipeline : &Vec<OperationWrapper>) -> Vec<f64> {
        let mut ret = Vec::new();
        for op in pipeline.iter() {
            match &op.op {
                Operation::Proba(targets) => {ret.push(self.proba(&targets));},
                _ => {op.apply(self.e, self.n, &mut self.state, &self.wiring);},
            }
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

    fn run(&mut self, pipeline : Vec<OperationWrapper>, ticks : usize) -> PyResult<Vec<f64>> {
        let mut ret = Vec::new();
        for _i in 0..ticks {
            for &x in self.apply(&pipeline).iter() {
                ret.push(x);
            }
        }
        Ok(ret)
    }

    fn reset(&mut self) {
        self.state = vec![Cplx::new(1./(2.*self.e as f64).sqrt(),0.);2*self.e];
    } 

    fn get_proba(&self, target : Vec<usize>) -> PyResult<f64> {
        Ok(self.proba(&target))
    }

    fn carac(&mut self, pipeline : Vec<OperationWrapper>, waiting : i32, timeout : usize) -> PyResult<(usize,f64)> {
        let mut min : f64 = -1.;
        let mut max : f64 = -1.;
        let mut pos : usize = 0;
        let mut steps : usize = 0;
        let mut waiting = waiting;
        let mut iter : usize = 0;

        loop {
            for &current in self.apply(&pipeline).iter() {
                steps+=1;
                if waiting <= 0 && current < (max+min)/2. {
                    return Ok((pos,max));
                }
                if current > max || steps == 1 {
                    max = current;
                    pos = steps;
                }
                if current < min || steps == 1 {
                    min = current;
                }
                waiting -= 1;
            }
            iter += 1;
            if iter > timeout {
                return Ok((pos,max));
            }
        }
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
    m.add_class::<UnitaryOp>()?;
    m.add_class::<OperationWrapper>()?;
    m.add_function(wrap_pyfunction!(_get_indices_around_nodes, m)?)?;
    Ok(())
}