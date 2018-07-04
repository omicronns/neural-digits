extern crate nalgebra as na;

use na::DMatrix;

mod mnist;

struct Network {
    layers: Vec<(DMatrix<f64>, DMatrix<f64>)>,
    activation: fn(f64) -> f64
}

impl Network {
    pub fn rand(layers_sizes: &[usize], activation: fn(f64) -> f64) -> Network {
        let mut layers = Vec::<(DMatrix<f64>, DMatrix<f64>)>::new();
        for sizes in layers_sizes.windows(2) {
            let wage = DMatrix::<f64>::new_random(sizes[1], sizes[0]);
            let bias = DMatrix::<f64>::new_random(sizes[1], 1);
            layers.push((wage, bias));
        }
        Network { layers, activation }
    }

    pub fn eval(&self, mut data: DMatrix<f64>) -> DMatrix<f64> {
        for (wage, bias) in &self.layers {
            println!("{:?}", wage.shape());
            println!("{:?}", bias.shape());
            data = wage * &data + bias;
        }
        data
    }
}

fn main() {
    let labels = mnist::import_data("./res/train-labels-idx1-ubyte.gz");
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data("./res/train-images-idx3-ubyte.gz");
    let images = mnist::Images::new(&images).unwrap();

    let x = images.get_flat(0).unwrap();

    let net = Network::rand(&[28 * 28, 15, 10], |x| x);

    for _ in 0..100 {
        let out = net.eval(x.clone());
        println!("{:?}",out);
    }
}
