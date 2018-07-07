extern crate nalgebra as na;

use self::na::DMatrix;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dsigmoid(x: f64) -> f64 {
    (-x).exp() / (1.0 + (-x).exp()).powi(2)
}

pub struct State {
    pub input: DMatrix<f64>,
    pub hidden: DMatrix<f64>,
    pub output: DMatrix<f64>,
}

impl State {
    pub fn predicted(&self) -> usize {
        self.output.iter().enumerate().scan((0, self.output[0]), |s, (it, x)| {
            *s = if *x > s.1 {
                (it, *x)
            } else {
                *s
            };
            Some(s.0)
        }).last().unwrap()
    }

    pub fn errors(&self, class: usize) -> DMatrix<f64> {
        DMatrix::<f64>::from_iterator(1, self.output.ncols(), self.output.iter().enumerate()
            .map(|(it, x)| if class == it { *x - 1.0 } else { *x }))
    }
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
    pub fn new_rand(input: usize, hidden: usize, output: usize, activation: fn(f64) -> f64, scale: f64) -> Self {
        Network {
            hidden: Layer {
                wages: DMatrix::<f64>::new_random(input, hidden).map(|x| x * scale - 0.5 * scale),
                bias: DMatrix::<f64>::new_random(1, hidden).map(|x| x * scale - 0.5 * scale)
            },
            output: Layer {
                wages: DMatrix::<f64>::new_random(hidden, output).map(|x| x * scale - 0.5 * scale),
                bias: DMatrix::<f64>::new_random(1, output).map(|x| x * scale - 0.5 * scale)
            },
            activation
        }
    }

    pub fn info(&self) {
        let hshape = self.hidden.wages.shape();
        let oshape = self.output.wages.shape();
        println!("++++++ Network info ++++++");
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
    pub rate: &'a Fn(usize) -> f64,
    pub data: &'a Fn(usize) -> Data,
    pub datalen: usize,
    pub dactivation: fn(f64) -> f64,
}

impl<'a> Trainer<'a> {
    pub fn new(net: Network, rate: &'a Fn(usize) -> f64, data: &'a Fn(usize) -> Data, datalen: usize, dactivation: fn(f64) -> f64) -> Trainer<'a> {
        Trainer {
            net, rate, data, datalen, dactivation
        }
    }

    pub fn learn(mut self, epochs: usize) -> Network {
        for epoch in 0..epochs {
            let rate = (self.rate)(epoch);
            println!("epoch: {}", epoch);
            println!("rate:  {}", rate);
            for it in 0..self.datalen {
                let data = (self.data)(it);
                let state = self.net.eval(data.data);
                let errors = state.errors(data.class);
                let derivatives = self.calc_derivatives(state, &errors);
                self.net.hidden.wages -= rate * &derivatives.dwagesh;
                self.net.hidden.bias -= rate * &derivatives.dbiash;
                self.net.output.wages -= rate * &derivatives.dwageso;
                self.net.output.bias -= rate * &derivatives.dbiaso;
                // println!("it: {:8} error: {}", it, errors.iter().map(|x| x.powi(2)).sum::<f64>());
            }
        }
        self.net
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
