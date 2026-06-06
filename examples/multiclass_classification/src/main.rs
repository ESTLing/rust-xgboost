use std::io::{BufRead, BufReader, BufWriter};
use std::fs::File;
use std::path::Path;
use xgboost_rs::{DMatrix, Booster};

fn main() {
    env_logger::init();

    download_dataset("dermatology.data");
    let (dtrain, dtest) = load_train_test_dmats("dermatology.data");
    let eval_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // Multiclass classification with softmax (6 classes)
    let bst = Booster::train(
        &[
            ("max_depth", "6"),
            ("eta", "0.1"),
            ("objective", "multi:softmax"),
            ("num_class", "6"),
            ("eval_metric", "merror"),
        ],
        &dtrain,
        5,
        Some(eval_sets),
    ).unwrap();

    let y_true = dtest.get_labels().unwrap();
    let y_pred = bst.predict(&dtest).unwrap();
    let num_errors = y_true.iter().zip(y_pred.iter())
        .filter(|(y1, y2)| y1 != y2)
        .count();
    println!("Test error using softmax: {}", num_errors as f32 / y_true.len() as f32);
}

fn download_dataset<P: AsRef<Path>>(dst: P) {
    let url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data";
    let dst = dst.as_ref();
    if dst.exists() {
        return;
    }
    let response = reqwest::blocking::get(url).expect("failed to download dataset");
    let file = File::create(dst).expect("failed to create file");
    let mut writer = BufWriter::new(file);
    response.copy_to(&mut writer).expect("failed to write file");
}

fn load_train_test_dmats<P: AsRef<Path>>(src: P) -> (DMatrix, DMatrix) {
    let src = src.as_ref();
    let file = File::open(src).expect("failed to open file");
    let reader = BufReader::new(file);

    let mut x: Vec<Vec<f32>> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let cols: Vec<f32> = line.split(',')
            .enumerate()
            .map(|(col_num, value)| match col_num {
                33 => if value == "?" { 1.0 } else { 0.0 },
                34 => value.parse::<f32>().unwrap() - 1.0,
                _ => value.parse::<f32>().unwrap(),
            })
            .collect();
        x.push(cols[0..33].to_vec());
        y.push(cols[34]);
    }

    let num_rows = x.len();
    let train_size = (0.7 * num_rows as f32) as usize;
    let x_train: Vec<f32> = x[0..train_size].iter().flat_map(|row| row.iter().cloned()).collect();
    let mut dtrain = DMatrix::from_dense(&x_train, train_size).unwrap();
    dtrain.set_labels(&y[0..train_size]).unwrap();

    let test_size = num_rows - train_size;
    let x_test: Vec<f32> = x[train_size..].iter().flat_map(|row| row.iter().cloned()).collect();
    let mut dtest = DMatrix::from_dense(&x_test, test_size).unwrap();
    dtest.set_labels(&y[train_size..]).unwrap();

    (dtrain, dtest)
}
