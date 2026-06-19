use xgboost_rs::{Booster, DMatrix, EarlyStopping, EvaluationMonitor, PredictConfig, KEY_GROUP, KEY_QID};

fn main() {
    // Synthetic ranking dataset: 3 queries, 10 documents, 2 features.
    //
    // Layout (row-major, 2 features per row):
    //   Query 1: 3 docs  → groups[0] = 3
    //   Query 2: 4 docs  → groups[1] = 4
    //   Query 3: 3 docs  → groups[2] = 3
    //
    // Feature values are crafted so that higher f0 roughly correlates with
    // higher relevance, while f1 adds controlled noise.
    let data: &[f32] = &[
        // Query 1 (3 docs)
        0.1, 0.3, // doc0 → label 0
        0.9, 0.6, // doc1 → label 2
        0.4, 0.5, // doc2 → label 1
        // Query 2 (4 docs)
        0.7, 0.8, // doc3 → label 2
        0.2, 0.4, // doc4 → label 0
        0.5, 0.6, // doc5 → label 1
        0.8, 0.7, // doc6 → label 2
        // Query 3 (3 docs)
        0.3, 0.5, // doc7 → label 1
        0.5, 0.2, // doc8 → label 0
        0.9, 0.9, // doc9 → label 2
    ];
    let nrows = 10;
    let ncols = 2;

    // Relevance labels (graded: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant)
    let labels: &[f32] = &[0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0];

    // Group sizes — one entry per query
    let groups: &[u32] = &[3, 4, 3];

    // Per-row query IDs (optional — group_ptr is the primary rank signal)
    let qids: &[u32] = &[0, 0, 0, 1, 1, 1, 1, 2, 2, 2];

    // ── Build training & eval matrices ────────────────────────────────
    let mut dtrain = DMatrix::from_dense(data, nrows).expect("from_dense dtrain");
    dtrain.set_label(labels).expect("set_label dtrain");
    dtrain.set_uint_info(KEY_GROUP, groups).expect("set_uint_info group");
    dtrain.set_uint_info(KEY_QID, qids).expect("set_uint_info qid");

    println!(
        "Train: {} docs × {} features, {} queries",
        dtrain.num_rows(),
        dtrain.num_cols(),
        groups.len()
    );

    // Use a training subset as eval (same pattern as Python demos)
    let eval_data = &data[..8 * ncols]; // first 8 docs
    let mut deval = DMatrix::from_dense(eval_data, 8).expect("from_dense deval");
    deval.set_label(&labels[..8]).expect("set_label deval");
    deval.set_uint_info(KEY_GROUP, &[3, 4, 1]).expect("set_uint_info group eval");
    deval.set_uint_info(KEY_QID, &[0, 0, 0, 1, 1, 1, 1, 2]).expect("set_uint_info qid eval");

    // ── Configure & train booster ─────────────────────────────────────
    let mut booster = Booster::new(ncols as usize).expect("Booster::new");

    booster
        .set_params(&[
            ("objective", "rank:ndcg"),
            ("eval_metric", "ndcg@3-"),
            ("max_depth", "3"),
            ("eta", "0.3"),
            ("lambdarank_num_pair_per_sample", "10"),
        ])
        .expect("set_params");

    booster.add_callback(Box::new(EvaluationMonitor::new(1)));
    booster.add_callback(Box::new(EarlyStopping::new(
        5,      // rounds without improvement
        "eval", // monitor this dataset
        "",     // last metric (ndcg@3-)
        true,   // maximize (NDCG is higher-is-better)
    )));

    let eval_sets = &[(&deval, "eval")];
    println!("\nTraining rank:ndcg for 50 rounds (early stopping enabled)...");
    let history = booster.train(&dtrain, 50, eval_sets).expect("train");

    // Show final NDCG
    if let Some(metrics) = history.get("eval") {
        if let Some(scores) = metrics.get("ndcg@3-") {
            println!("\nFinal NDCG@3: {:.4} ({} iterations)", scores.last().unwrap(), scores.len());
        }
    }

    // ── Predict ───────────────────────────────────────────────────────
    let preds = booster.predict(&deval, &PredictConfig::default()).expect("predict");

    println!("\nPer-document scores (higher = more relevant):");
    let mut off = 0;
    for (q, &g) in groups.iter().enumerate() {
        let g = g as usize;
        print!("  Query {}: [", q);
        for i in off..off + g {
            if i >= preds.len() {
                break;
            }
            print!("{:.4}", preds[i]);
            if i < off + g - 1 {
                print!(", ");
            }
        }
        println!("]");
        off += g;
    }
}
