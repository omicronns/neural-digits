#[macro_use]
extern crate serde_derive;

mod mnist;
mod nnet;

fn train(labels_path: &'static str, images_path: &'static str, netpath: Option<&str>) -> nnet::Network {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    let net = match netpath {
        Some(path) => match nnet::load_net(path) {
            Some(net) => {
                println!("network loaded from: {}", path);
                net
            },
            None => nnet::Network::new_rand(&[images.size.0 * images.size.1, 15, 10], 10.0)
        },
        None => nnet::Network::new_rand(&[images.size.0 * images.size.1, 15, 10], 10.0)
    };
    net.info();

    let epochs = 30;
    let data = |n| nnet::Data { class: labels[n] as usize, data: images.get_flat(n).unwrap() };
    let rate = |e| 3.0 - e as f64 * (2.0 / epochs as f64);
    let trainer = nnet::Trainer::new(net, &rate, &data, 10000);

    trainer.learn(epochs)
}

fn check(net: nnet::Network, labels_path: &'static str, images_path: &'static str) {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    let mut successes = 0;
    for it in 0..1000 {
        let img = images.get_flat(it).unwrap();
        let state = net.eval(img);
        let expected = labels[it] as usize;
        if state.class() == expected {
            successes += 1;
        }
    }
    println!("success rate: {}", successes as f64 / 1000.0);
}

fn main() {
    let net_trained = train("./res/train-labels-idx1-ubyte.gz", "./res/train-images-idx3-ubyte.gz", Some("./res/netfile.bin"));
    check(net_trained, "./res/train-labels-idx1-ubyte.gz", "./res/train-images-idx3-ubyte.gz");
}
