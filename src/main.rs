#[macro_use] extern crate serde_derive;

mod mnist;
mod nnet;

fn read<T>() -> Result<T, String>
where
    T: std::str::FromStr {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let line = input.trim_right();
    match line.parse() {
        Ok(n) => Ok(n),
        Err(_) => Err(String::from(line)),
    }
}

fn train(labels_path: &'static str, images_path: &'static str, netpath: Option<&str>) -> nnet::Network {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    let epochs: usize;
    println!("configure network:");
    loop {
        println!("epochs:");
        epochs = match read::<usize>() {
            Ok(n) => n,
            Err(s) => {
                println!("invalid input");
                continue;
            }
        };
        break;
    }
    let mut dims = Vec::<usize>::new();
    dims.push(images.size.0 * images.size.1);
    println!("hint: one 15 neurons layer is a good choice");
    loop {
        println!("add layer, select number of neurons or q to stop:");
        dims.push(match read::<usize>() {
            Ok(n) => n,
            Err(s) => match &s[..] {
                "q" => break,
                _ => {
                    println!("invalid input");
                    continue;
                }
            }
        });
    }
    dims.push(10);
    let net = match netpath {
        Some(netpath) => match nnet::Network::from_file(netpath) {
            Some(net) => {
                println!("network loaded from: {}", netpath);
                net
            },
            None => nnet::Network::new_rand(&dims, 10.0)
        },
        None => nnet::Network::new_rand(&dims, 10.0)
    };
    net.info();

    let data = |n| nnet::Data { class: labels[n] as usize, data: images.get_flat(n).unwrap() };
    let rate = |e| 3.0 - e as f64 * (2.0 / epochs as f64);
    let trainer = nnet::Trainer::new(net, &rate, &data, 10000);

    trainer.learn(epochs)
}

fn check(net: &nnet::Network, labels_path: &'static str, images_path: &'static str) {
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

fn check_manual(net: &nnet::Network, labels_path: &'static str, images_path: &'static str) {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    let mut successes = 0;
    loop {
        println!("select image number or q to quit [0-{}]:", labels.len() - 1);
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let n = match input.trim_right().parse::<usize>() {
            Ok(n) => n,
            Err(_) => match &input[..] {
                "q\n" => break,
                _ => {
                    println!("invalid input");
                    continue;
                }
            },
        };
        let img = images.get_flat(n).unwrap();
        nnet::print_matrix(28, 28, &img);
        let state = net.eval(img);
        let expected = labels[n] as usize;
        let prediction = state.class();
        if prediction == expected {
            println!("predicted {} right", prediction);
            successes += 1;
        } else {
            println!("predicted {} wrong, expected {}", prediction, expected);
        }
    }
    println!("success rate: {}", successes as f64 / 1000.0);
}

fn main() {
    let net_trained = train("./res/train-labels-idx1-ubyte.gz", "./res/train-images-idx3-ubyte.gz", Some("./res/netfile.bin"));
    check(&net_trained, "./res/t10k-labels-idx1-ubyte.gz", "./res/t10k-images-idx3-ubyte.gz");
    check_manual(&net_trained, "./res/t10k-labels-idx1-ubyte.gz", "./res/t10k-images-idx3-ubyte.gz");
}
