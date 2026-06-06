use xgboost_rs::{Booster, DMatrix};

fn main() {
    // Create a simple synthetic dataset: 8 rows, 3 features
    let data = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 1.2, 2.2, 3.2, 4.2,
        5.2, 6.2,
    ];
    let num_rows = 8;
    let mut dtrain = DMatrix::from_dense(data, num_rows)
        .expect("from_dense dtrain");
    dtrain.set_labels(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        .expect("set_labels dtrain");
    println!("Train matrix: {}x{}", dtrain.num_rows(), dtrain.num_cols());

    let mut dtest = DMatrix::from_dense(data, num_rows)
        .expect("from_dense dtest");
    dtest.set_labels(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        .expect("set_labels dtest");

    // Train using flat string key-value parameters
    let eval_sets = &[(&dtest, "test"), (&dtrain, "train")];
    println!("\nTraining tree booster...");
    let booster = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dtrain,
        10,
        Some(eval_sets),
    )
    .expect("train");

    // Predict
    let preds = booster.predict(&dtest).expect("predict");
    println!("Predictions: {:?}", &preds[..4]);

    // Save and load
    println!("\nSaving and loading Booster model...");
    booster.save("xgb.json").expect("save booster");
    let booster2 = Booster::load("xgb.json").expect("load booster");
    let preds2 = booster2.predict(&dtest).expect("predict after load");
    assert_eq!(preds, preds2);

    // Save and load DMatrix
    println!("\nSaving and loading matrix data...");
    dtest.save("test.dmat").expect("save dmatrix");
    let dtest2 = DMatrix::load_binary("test.dmat").expect("load_binary dtest");
    assert_eq!(booster.predict(&dtest2).expect("predict on loaded dmat"), preds);

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
    let mut dmat = DMatrix::from_csr(indptr, indices, sparse_data, Some(3))
        .expect("from_csr");
    dmat.set_labels(&[0.0, 1.0, 0.0]).expect("set_labels csr");
    let bst = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dmat,
        2,
        None,
    )
    .expect("train csr");
    println!("CSR predictions: {:?}", bst.predict(&dmat).expect("predict csr"));
}
