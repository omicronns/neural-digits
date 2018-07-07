extern crate nalgebra as na;

use self::na::DMatrix;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dsigmoid(x: f64) -> f64 {
    (-x).exp() / (1.0 + (-x).exp()).exp2()
}

pub struct State {
    pub input: DMatrix<f64>,
    pub hidden: DMatrix<f64>,
    pub output: DMatrix<f64>,
}

pub struct Layer {
    pub wages: DMatrix<f64>,
    pub bias: DMatrix<f64>,
}

pub struct Network {
    pub hidden: Layer,
    pub output: Layer,
    pub activation: fn(f64) -> f64,
}

impl Network {
    pub fn new_rand(input: usize, hidden: usize, output: usize, activation: fn(f64) -> f64) -> Self {
        let hidden_layer = Layer {
            wages: DMatrix::<f64>::new_random(input, hidden),
            bias: DMatrix::<f64>::new_random(1, hidden)
        };
        let output_layer = Layer {
            wages: DMatrix::<f64>::new_random(hidden, output),
            bias: DMatrix::<f64>::new_random(1, output)
        };
        Network { hidden: hidden_layer, output: output_layer, activation }
    }

    pub fn info(&self) {
        println!("++++++ Network info ++++++");
        let hshape = self.hidden.wages.shape();
        let oshape = self.output.wages.shape();
        println!("input dim:         {}", hshape.0);
        println!("hidden layer dims: ({},{})", hshape.0, hshape.1);
        println!("output layer dims: ({},{})", oshape.0, oshape.1);
        println!("output dim:        {}", oshape.1);
        println!("++++++++++++++++++++++++++");
    }

    pub fn eval(&self, input: DMatrix<f64>) -> State {
        let hidden = (&input * &self.hidden.wages + &self.hidden.bias).map(|x| (self.activation)(x));
        let output = (&hidden * &self.output.wages + &self.output.bias).map(|x| (self.activation)(x));
        State {input, hidden, output }
    }
}

pub struct Data {
    pub class: usize,
    pub data: DMatrix<f64>,
}

struct Derivatives {
    dbiaso: DMatrix<f64>,
    dbiash: DMatrix<f64>,
    dwageso: DMatrix<f64>,
    dwagesh: DMatrix<f64>,
}

pub struct Trainer<'a> {
    pub net: Network,
    pub rate: f64,
    pub data: &'a Fn(usize) -> Data,
    pub datalen: usize,
    pub dactivation: fn(f64) -> f64,
}

impl<'a> Trainer<'a> {
    pub fn new(net: Network, rate: f64, data: &'a Fn(usize) -> Data, datalen: usize, dactivation: fn(f64) -> f64) -> Trainer<'a> {
        Trainer {
            net, rate, data, datalen, dactivation
        }
    }

    pub fn learn(mut self) -> Network {
        for epoch in 0..self.datalen {
            println!("++++++ epoch begin: {}", epoch);
            let data = (self.data)(epoch);
            let state = self.net.eval(data.data);
            let errors = self.calc_errors(&state.output, data.class);
            let derivatives = self.calc_derivatives(state, &errors);
            self.net.hidden.wages -= self.rate * &derivatives.dwagesh;
            self.net.hidden.bias -= self.rate * &derivatives.dbiash;
            self.net.output.wages -= self.rate * &derivatives.dwageso;
            self.net.output.bias -= self.rate * &derivatives.dbiaso;
            println!("-- error: {}", errors.iter().map(|x| x.powi(2)).sum::<f64>());
            println!("++++++ epoch end: {}", epoch);
            println!("");
        }
        self.net
    }

    fn calc_errors(&self, output: &DMatrix<f64>, class: usize) -> DMatrix<f64> {
        DMatrix::<f64>::from_iterator(1, output.ncols(), output.iter().enumerate()
            .map(|(it, x)| if class == it {
                *x - 1.0
            } else {
                *x
            }))
    }

    fn calc_derivatives(&self, state: State, errors: &DMatrix<f64>) -> Derivatives {
        let dbiaso = (&state.hidden * &self.net.output.wages + &self.net.output.bias)
            .map(|x| (self.dactivation)(x))
            .component_mul(&errors);
        let dbiash = (&state.input * &self.net.hidden.wages + &self.net.hidden.bias)
            .map(|x| (self.dactivation)(x))
            .component_mul(&(&dbiaso * &self.net.output.wages.transpose()));
        let dwageso = &state.hidden.transpose() * &dbiaso;
        let dwagesh = &state.input.transpose() * &dbiash;
        Derivatives {
            dbiaso, dbiash, dwageso, dwagesh
        }
    }
}
