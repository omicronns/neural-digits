extern crate serde_derive;
extern crate bincode;
extern crate nalgebra as na;

use self::bincode::{serialize_into, deserialize_from};
use self::na::DMatrix;

use std::fs::File;

pub fn load_net(path: &str) -> Network {
    let netfile = File::open(path).expect("could not open file");
    deserialize_from(&netfile).expect("could not deserialize")
}

pub fn dump_net(net: &Network, path: &str) {
    let mut netfile = File::create(path).expect("could not create file");
    serialize_into(&mut netfile, &net).expect("could not serialize");
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn dsigmoid(x: f64) -> f64 {
    (-x).exp() / (1.0 + (-x).exp()).powi(2)
}

pub struct State {
    pub layers: Vec<DMatrix<f64>>,
}

impl State {
    pub fn class(&self) -> usize {
        let output = self.layers.iter().last().unwrap();
        let first = output[0];
        output.iter()
            .enumerate()
            .scan((0, first), |s, (it, x)| {
            *s = if *x > s.1 {
                (it, *x)
            } else {
                *s
            };
            Some(s.0)
        }).last().unwrap()
    }

    pub fn errors(&self, class: usize) -> DMatrix<f64> {
        let output = self.layers.iter().last().unwrap();
        DMatrix::<f64>::from_iterator(1, output.ncols(), output.iter().enumerate()
            .map(|(it, x)| if class == it { *x - 1.0 } else { *x }))
    }
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub wages: DMatrix<f64>,
    pub bias: DMatrix<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new_rand(dims: &[usize], scale: f64) -> Self {
        let mut layers = Vec::<Layer>::new();
        for dim in dims.windows(2) {
            layers.push(Layer {
                wages: DMatrix::<f64>::new_random(dim[0], dim[1]).map(|x| x * scale - 0.5 * scale),
                bias: DMatrix::<f64>::new_random(1, dim[1]).map(|x| x * scale - 0.5 * scale)
            });
        }
        Network {
            layers
        }
    }

    pub fn info(&self) {
        println!("++++++ Network info ++++++");
        for (it, layer) in self.layers.iter().enumerate() {
            println!("layer {:2} dim({},{})", it, layer.wages.nrows(), layer.wages.ncols());
        }
        println!("++++++++++++++++++++++++++");
    }

    pub fn eval(&self, input: DMatrix<f64>) -> State {
        let mut layers = Vec::<DMatrix<f64>>::new();
        let mut state = input.clone();
        layers.push(input);
        for layer in &self.layers {
            state = (&state * &layer.wages + &layer.bias).map(|x| sigmoid(x));
            layers.push(state.clone());
        }
        State { layers }
    }
}

pub struct Data {
    pub class: usize,
    pub data: DMatrix<f64>,
}

pub struct Trainer<'a> {
    pub net: Network,
    pub rate: &'a Fn(usize) -> f64,
    pub data: &'a Fn(usize) -> Data,
    pub datalen: usize,
}

impl<'a> Trainer<'a> {
    pub fn new(net: Network, rate: &'a Fn(usize) -> f64, data: &'a Fn(usize) -> Data, datalen: usize) -> Trainer<'a> {
        Trainer {
            net, rate, data, datalen
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
                let mut errors = state.errors(data.class);
                let mut wages = DMatrix::<f64>::identity(errors.ncols(), errors.ncols());
                for (layer, state) in self.net.layers.iter_mut().rev().zip(state.layers.iter().rev().skip(1)) {
                    let deirvative = (state * &layer.wages + &layer.bias).map(|x| dsigmoid(x));
                    let dbias = (&errors * &wages).component_mul(&deirvative);
                    let dwages = state.transpose() * &dbias;
                    layer.wages -= rate * dwages;
                    layer.bias -= rate * &dbias;
                    errors = dbias;
                    wages = layer.wages.transpose();
                }
                let error = errors.iter().map(|x| x.powi(2)).sum::<f64>();
                println!("it: {:8} error: {:.16}", it, error);
            }
        }
        self.net
    }
}
