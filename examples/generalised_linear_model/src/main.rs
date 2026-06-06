use xgboost_rs::{DMatrix, Booster};

fn main() {
    env_logger::init();

    // Synthetic dataset
    let data = &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        1.1, 2.1, 3.1,
        4.1, 5.1, 6.1,
        7.1, 8.1, 9.1,
        1.2, 2.2, 3.2,
        4.2, 5.2, 6.2,
    ];
    let num_rows = 8;
    let mut dtrain = DMatrix::from_dense(data, num_rows).unwrap();
    dtrain.set_labels(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();

    let mut dtest = DMatrix::from_dense(data, num_rows).unwrap();
    dtest.set_labels(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();

    let eval_sets = &[(&dtest, "test"), (&dtrain, "train")];

    // Linear booster (generalised linear model)
    println!("\nTraining linear booster...");
    let bst = Booster::train(
        &[
            ("booster", "gblinear"),
            ("alpha", "0.0001"),
            ("lambda", "1.0"),
            ("objective", "binary:logistic"),
        ],
        &dtrain,
        4,
        Some(eval_sets),
    ).unwrap();

    let preds = bst.predict(&dtest).unwrap();
    let labels = dtest.get_labels().unwrap();
    let num_errors = preds.iter().zip(labels.iter())
        .filter(|(p, l)| (*p > 0.5) as u8 as f32 != *l)
        .count();
    println!("error={} ({}/{} correct)",
             num_errors as f32 / preds.len() as f32,
             preds.len() - num_errors,
             preds.len());
}
