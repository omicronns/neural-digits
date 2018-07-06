extern crate nalgebra as na;

mod mnist;
mod nnet;

fn main() {
    let labels = mnist::import_data("./res/train-labels-idx1-ubyte.gz");
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data("./res/train-images-idx3-ubyte.gz");
    let images = mnist::Images::new(&images).unwrap();

    let img = images.get_flat(0).unwrap();
    let net = nnet::Network::new_rand(28 * 28, 15, 10, nnet::sigmoid);
    let state = net.eval(img);

    net.info();

    let data = |n| nnet::Data { class: labels[n as usize].clone() as usize, data: images.get_flat(n).unwrap() };
    let trainer = nnet::Trainer::new(net, 1.0, &data, nnet::dsigmoid);
    let dx = trainer.calc_derivatives(state, 0);
    println!("{:?}", dx);
    // for _ in 0..100 {
    //     let out = net.eval(x.clone());
    //     println!("{:?}",out);
    // }
}
