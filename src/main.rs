#[macro_use] extern crate serde_derive;

mod mnist;
mod nnet;

fn read<T>() -> Result<T, String>
where
    T: std::str::FromStr {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let line = input.trim();
    match line.parse() {
        Ok(n) => Ok(n),
        Err(_) => Err(String::from(line)),
    }
}

fn get_setup(net: Option<nnet::Network>, inputs: usize) -> (nnet::Network, Option<usize>, Option<usize>) {
    println!("configure network:");
    println!("datalen:");
    let datalen = match read::<usize>() {
        Ok(n) => Some(n),
        Err(_) => {
            println!("defaulting");
            None
        }
    };
    match net {
        Some(net) => (net, datalen, None),
        None => {
            let epochs;
            loop {
                println!("epochs:");
                epochs = match read::<usize>() {
                    Ok(n) => Some(n),
                    Err(_) => {
                        println!("invalid input");
                        continue;
                    }
                };
                break;
            }
            let mut dims = Vec::<usize>::new();
            dims.push(inputs);
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
            (nnet::Network::new_rand(&dims, 10.0), datalen, epochs)
        }
    }
}

fn train(labels_path: &'static str, images_path: &'static str, netpath: Option<&str>) -> nnet::Network {
    let labels = mnist::import_data(labels_path);
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data(images_path);
    let images = mnist::Images::new(&images).unwrap();

    let setup = match netpath {
        Some(netpath) => match nnet::Network::from_file(netpath) {
            Some(net) => {
                println!("network loaded from: {}", netpath);
                get_setup(Some(net), images.size.0 * images.size.1)
            },
            None => get_setup(None, images.size.0 * images.size.1)
        },
        None => get_setup(None, images.size.0 * images.size.1)
    };

    let net = match setup {
        (net, datalen, Some(epochs)) => {
            let data = |n| nnet::Data { class: labels[n] as usize, data: images.get_flat(n).unwrap() };
            let rate = |e| 3.0 - e as f64 * (2.0 / epochs as f64);

            let trainer = match datalen {
                Some(len) => nnet::Trainer::new(net, &rate, &data, len),
                None => nnet::Trainer::new(net, &rate, &data, labels.len())
            };
            trainer.learn(epochs)
        },
        (net, _, _) => net
    };
    net.info();
    net
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

    let mut successes = 0.0;
    let mut it = 0.0;
    loop {
        println!("select image number, d to dump or q to quit [0-{}]:", labels.len() - 1);
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let n = match input.trim_right().parse::<usize>() {
            Ok(n) => n,
            Err(_) => match &input[..] {
                "q\n" => break,
                "d\n" => {
                    net.dump("./res/net.bin");
                    continue;
                }
                _ => {
                    println!("invalid input");
                    continue;
                }
            },
        };
        let img = images.get_flat(n).unwrap();
        nnet::print_matrix(images.size.0, images.size.1, &img);
        let state = net.eval(img);
        let expected = labels[n] as usize;
        let prediction = state.class();
        it += 1.0;
        if prediction == expected {
            println!("predicted {} right", prediction);
            successes += 1.0;
        } else {
            println!("predicted {} wrong, expected {}", prediction, expected);
        }
    }
    println!("success rate: {}", successes / it);
}

fn main() {
    let net = train("./res/train-labels-idx1-ubyte.gz", "./res/train-images-idx3-ubyte.gz", Some("./res/net.bin"));
    check(&net, "./res/t10k-labels-idx1-ubyte.gz", "./res/t10k-images-idx3-ubyte.gz");
    check_manual(&net, "./res/t10k-labels-idx1-ubyte.gz", "./res/t10k-images-idx3-ubyte.gz");
}
