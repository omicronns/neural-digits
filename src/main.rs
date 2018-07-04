mod mnist;

fn main() {
    let labels = mnist::import_data("./res/train-labels-idx1-ubyte.gz");
    let labels = mnist::get_labels(&labels).unwrap();
    let images = mnist::import_data("./res/train-images-idx3-ubyte.gz");
    let images = mnist::Images::new(&images).unwrap();
    let img0 = images.get(0).unwrap();
    let img1 = images.get(1).unwrap();
    let img = img1;

    for itr in 0..img.nrows() {
        for itc in 0..img.ncols() {
            print!("{:02x}", img[itr * img.ncols() + itc]);
        }
        println!("");
    }
}
