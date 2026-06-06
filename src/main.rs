use xgboost_rs::{Booster, DMatrix};

fn main() {
    // Create a simple synthetic dataset: 8 rows, 3 features
    let data = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 1.2, 2.2, 3.2, 4.2,
        5.2, 6.2,
    ];
    let num_rows = 8;
    let mut dtrain = DMatrix::from_dense(data, num_rows).unwrap();
    dtrain.set_labels(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
    println!("Train matrix: {}x{}", dtrain.num_rows(), dtrain.num_cols());

    let mut dtest = DMatrix::from_dense(data, num_rows).unwrap();
    dtest.set_labels(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();

    // Train using flat string key-value parameters (like Python's xgb.train)
    let eval_sets = &[(&dtest, "test"), (&dtrain, "train")];
    println!("\nTraining tree booster...");
    let booster = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dtrain,
        10,
        Some(eval_sets),
    )
    .unwrap();

    // Predict
    let preds = booster.predict(&dtest).unwrap();
    println!("Predictions: {:?}", &preds[..4]);

    // Save and load
    println!("\nSaving and loading Booster model...");
    booster.save("xgb.json").unwrap();
    let booster2 = Booster::load("xgb.json").unwrap();
    let preds2 = booster2.predict(&dtest).unwrap();
    assert_eq!(preds, preds2);

    // Save and load DMatrix
    println!("\nSaving and loading matrix data...");
    dtest.save("test.dmat").unwrap();
    let dtest2 = DMatrix::load_binary("test.dmat").unwrap();
    assert_eq!(booster.predict(&dtest2).unwrap(), preds);

    // Error handling
    println!("\nError message example...");
    match Booster::load("/does/not/exist") {
        Err(err) => println!("Got expected error: {}", err),
        _ => (),
    }

    // Sparse matrix (CSR)
    println!("\nSparse matrix construction...");
    let indptr = &[0, 2, 3, 4];
    let indices = &[0, 2, 2, 1];
    let sparse_data = &[1.0, 2.0, 3.0, 4.0];
    let mut dmat = DMatrix::from_csr(indptr, indices, sparse_data, Some(3)).unwrap();
    dmat.set_labels(&[0.0, 1.0, 0.0]).unwrap();
    let bst = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dmat,
        2,
        None,
    )
    .unwrap();
    println!("CSR predictions: {:?}", bst.predict(&dmat).unwrap());
}
