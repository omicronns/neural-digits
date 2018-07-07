mod mnist;
mod nnet;

fn train(labels_path: &'static str, images_path: &'static str) -> nnet::Network {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    let net = nnet::Network::new_rand(28 * 28, 15, 10, nnet::sigmoid);
    net.info();

    let data = |n| nnet::Data { class: labels[n as usize].clone() as usize, data: images.get_flat(n).unwrap() };
    let trainer = nnet::Trainer::new(net, 10.0, &data, 10000, nnet::dsigmoid);

    trainer.learn()
}

fn check(net: nnet::Network, labels_path: &'static str, images_path: &'static str) {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    for it in 0..labels.len() {
        let img = images.get_flat(it).unwrap();
        let prediction = net.eval(img);
        let expected = labels[it] as usize;
        let hit = prediction.output.iter().enumerate()
            .all(|(class, x)| if class == expected { *x > 0.5 } else { *x < 0.5 });
        if hit {
            println!("prediction valid, yay!");
        } else {
            println!("prediction invalid");
        }
    }
}

fn main() {
    let net_trained = train("./res/train-labels-idx1-ubyte.gz", "./res/train-images-idx3-ubyte.gz");
    check(net_trained, "./res/train-labels-idx1-ubyte.gz", "./res/train-images-idx3-ubyte.gz");
}
