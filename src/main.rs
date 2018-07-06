extern crate nalgebra as na;

mod mnist;
mod nnet;

fn main() {
    let labels = mnist::import_data("./res/train-labels-idx1-ubyte.gz");
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data("./res/train-images-idx3-ubyte.gz");
    let images = mnist::Images::new(&images).unwrap();

    let x = images.get_flat(0).unwrap();

    let net = nnet::Network::rand(28 * 28, 15, 10, nnet::sigmoid);

    let out = net.eval(x.clone());
    // println!("{:?}", Network::errors(out, labels, 1));
    // println!("{:?}", net.layers[0].0);
    println!("{:?}", net.valid());

    net.info();
    // for _ in 0..100 {
    //     let out = net.eval(x.clone());
    //     println!("{:?}",out);
    // }
}
