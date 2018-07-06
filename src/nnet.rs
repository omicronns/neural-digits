extern crate nalgebra as na;

use self::na::DMatrix;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dsigmoid(x: f64) -> f64 {
    (-x).exp() / (1.0 + (-x).exp()).exp2()
}

pub struct State {
    input: DMatrix<f64>,
    hidden: DMatrix<f64>,
    output: DMatrix<f64>,
}

pub struct Layer {
    wages: DMatrix<f64>,
    bias: DMatrix<f64>,
}

pub struct Network {
    hidden: Layer,
    output: Layer,
    activation: fn(f64) -> f64,
}

impl Network {
    pub fn new_rand(input: usize, hidden: usize, output: usize, activation: fn(f64) -> f64) -> Network {
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
        println!("input dim:         {}", hshape.0);
        println!("++++++++++++++++++++++++++");
    }

    pub fn valid(&self) -> Result<(), &'static str> {
        if self.hidden.wages.ncols() != self.output.wages.nrows() {
            return Err("invalid wage matrices dimentions");
        }
        if self.hidden.wages.ncols() != self.hidden.bias.ncols() ||
           self.output.wages.ncols() != self.output.bias.ncols() {
            return Err("invalid bias size");
        }
        Ok(())
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

pub struct Derivatives {
    djdwage: DMatrix<f64>,
    djdbias: DMatrix<f64>,
}

pub struct Trainer<'a> {
    net: Network,
    rate: f64,
    data: &'a Fn(usize) -> Data,
    dactivation: fn(f64) -> f64,
}

impl<'a> Trainer<'a> {
    pub fn new(net: Network, rate: f64, data: &'a Fn(usize) -> Data, dactivation: fn(f64) -> f64) -> Trainer<'a> {
        Trainer {
            net, rate, data, dactivation
        }
    }

    pub fn detach(self) -> Network {
        self.net
    }

    pub fn calc_errors(&self, output: DMatrix<f64>, n: usize) -> DMatrix<f64> {
        DMatrix::<f64>::from_iterator(1, output.ncols(), output.iter().enumerate()
            .map(|(it, x)| if it == n {
                1.0 - *x
            } else {
                *x
            }))
    }

    pub fn calc_derivatives(&self, state: State, n: usize) {
        let errors = self.calc_errors(state.output, n);
        let sigma2 = errors * (state.hidden * &self.net.hidden.wages + &self.net.hidden.bias).map(|x| (self.dactivation)(x));
        println!("{:?}", sigma2.shape());
    }
}
