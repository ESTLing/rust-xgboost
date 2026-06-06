# Maintenance Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the codebase into a safe, maintainable, publishable Rust XGBoost binding following the design spec.

**Architecture:** Four sequential phases — safety fixes first (foundation), then build.rs rewrite (PyPI-based), then API simplification (delete all parameter structs, flat `&[(&str, &str)]` params like Python), then engineering polish (version/CI/CHANGELOG).

**Tech Stack:** Rust 2021 edition, bindgen 0.71, ureq, zip, serde

---

## Phase 1: Safety Fixes

### Task 1.1: Fix Drop implementations to not panic

**Files:**
- Modify: `src/booster.rs:733-737`
- Modify: `src/dmatrix.rs:360-364`

- [ ] **Step 1: Fix Booster::drop**

Replace lines 733-737 in `src/booster.rs`:

```rust
impl Drop for Booster {
    fn drop(&mut self) {
        if let Err(e) = xgb_call!(xgboost_sys::XGBoosterFree(self.handle)) {
            error!("XGBoosterFree failed in drop: {}", e);
        }
    }
}
```

- [ ] **Step 2: Fix DMatrix::drop**

Replace lines 360-364 in `src/dmatrix.rs`:

```rust
impl Drop for DMatrix {
    fn drop(&mut self) {
        if let Err(e) = xgb_call!(xgboost_sys::XGDMatrixFree(self.handle)) {
            error!("XGDMatrixFree failed in drop: {}", e);
        }
    }
}
```

- [ ] **Step 3: Build to verify changes compile**

Run: `cargo build --verbose`
Expected: Compiles successfully.

- [ ] **Step 4: Commit**

```bash
git add src/booster.rs src/dmatrix.rs
git commit -m "fix: remove unwrap from Drop impls to prevent double-panic"
```

---

### Task 1.2: Replace release-unsafe assertions with proper error handling

**Files:**
- Modify: `src/booster.rs:486-487` (predict_matrix)
- Modify: `src/booster.rs:517-518` (predict)
- Modify: `src/booster.rs:539-540` (predict_margin)
- Modify: `src/booster.rs:563-564` (predict_leaf)
- Modify: `src/booster.rs:592-593` (predict_contributions)
- Modify: `src/booster.rs:622-623` (predict_interactions)

- [ ] **Step 1: Fix predict_matrix (line 487)**

Replace:
```rust
        assert!(!out_result.is_null());
```
with:
```rust
        if out_result.is_null() {
            return Err(XGBError::new("predict_matrix: null result pointer".to_string()));
        }
```

- [ ] **Step 2: Fix predict (line 517)**

Replace:
```rust
        assert!(!out_result.is_null());
```
with:
```rust
        if out_result.is_null() {
            return Err(XGBError::new("predict: null result pointer".to_string()));
        }
```

- [ ] **Step 3: Fix predict_margin (line 539)**

Replace:
```rust
        assert!(!out_result.is_null());
```
with:
```rust
        if out_result.is_null() {
            return Err(XGBError::new("predict_margin: null result pointer".to_string()));
        }
```

- [ ] **Step 4: Fix predict_leaf (line 563)**

Replace:
```rust
        assert!(!out_result.is_null());
```
with:
```rust
        if out_result.is_null() {
            return Err(XGBError::new("predict_leaf: null result pointer".to_string()));
        }
```

- [ ] **Step 5: Fix predict_contributions (line 592)**

Replace:
```rust
        assert!(!out_result.is_null());
```
with:
```rust
        if out_result.is_null() {
            return Err(XGBError::new("predict_contributions: null result pointer".to_string()));
        }
```

- [ ] **Step 6: Fix predict_interactions (line 622)**

Replace:
```rust
        assert!(!out_result.is_null());
```
with:
```rust
        if out_result.is_null() {
            return Err(XGBError::new("predict_interactions: null result pointer".to_string()));
        }
```

- [ ] **Step 7: Build and test**

Run: `cargo build --verbose && cargo test --verbose`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/booster.rs
git commit -m "fix: replace assert! with proper error returns for null pointers"
```

---

### Task 1.3: Fix UTF-8 unwrap calls that can panic

**Files:**
- Modify: `src/booster.rs:346` (eval_set CStr conversion)
- Modify: `src/booster.rs:383` (get_attribute CStr)
- Modify: `src/booster.rs:403` (get_attribute_names)
- Modify: `src/booster.rs:432` (get_feature_info)
- Modify: `src/booster.rs:688` (dump_model)

- [ ] **Step 1: Fix eval_set (line 346)**

Replace:
```rust
        let out = unsafe { ffi::CStr::from_ptr(out_result).to_str().unwrap().to_owned() };
```
with:
```rust
        let out = unsafe {
            ffi::CStr::from_ptr(out_result)
                .to_str()
                .map_err(|e| XGBError::new(format!("eval output not valid UTF-8: {}", e)))?
                .to_owned()
        };
```

- [ ] **Step 2: Fix get_attribute (line 383)**

Replace:
```rust
        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str().unwrap();
        Ok(Some(out.to_owned()))
```
with:
```rust
        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str()
            .map_err(|e| XGBError::new(format!("attribute not valid UTF-8: {}", e)))?;
        Ok(Some(out.to_owned()))
```

- [ ] **Step 3: Fix get_attribute_names (line 403)**

Replace:
```rust
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
                .collect();
```
with:
```rust
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe {
                    ffi::CStr::from_ptr(*str_ptr)
                        .to_str()
                        .map(|s| s.to_owned())
                        .map_err(|e| XGBError::new(format!("attribute name not valid UTF-8: {}", e)))
                })
                .collect::<XGBResult<Vec<String>>>()?;
```

- [ ] **Step 4: Fix get_feature_info (line 432)**

Replace:
```rust
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
                .collect();
```
with:
```rust
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe {
                    ffi::CStr::from_ptr(*str_ptr)
                        .to_str()
                        .map(|s| s.to_owned())
                        .map_err(|e| XGBError::new(format!("feature info not valid UTF-8: {}", e)))
                })
                .collect::<XGBResult<Vec<String>>>()?;
```

- [ ] **Step 5: Fix dump_model (line 688)**

Replace:
```rust
            let out_vec: Vec<String> = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
                .collect();
```
with:
```rust
            let out_vec: Vec<String> = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe {
                    ffi::CStr::from_ptr(*str_ptr)
                        .to_str()
                        .map(|s| s.to_owned())
                        .map_err(|e| XGBError::new(format!("dump model text not valid UTF-8: {}", e)))
                })
                .collect::<XGBResult<Vec<String>>>()?;
```

- [ ] **Step 6: Build and test**

Run: `cargo build --verbose && cargo test --verbose`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/booster.rs
git commit -m "fix: replace CStr::to_str().unwrap() with proper error handling"
```

---

### Task 1.4: Add From impl for UTF-8 conversion errors

**Files:**
- Modify: `src/error.rs`

- [ ] **Step 1: Add FromStrUtf8Error conversion**

After the existing `impl Display for XGBError` block (around line 51), add:

```rust
impl From<std::str::Utf8Error> for XGBError {
    fn from(e: std::str::Utf8Error) -> Self {
        XGBError::new(format!("UTF-8 conversion error: {}", e))
    }
}
```

This allows `?` to work directly with `to_str()` results.

- [ ] **Step 2: Build to verify**

Run: `cargo build --verbose`
Expected: Compiles successfully.

- [ ] **Step 3: Commit**

```bash
git add src/error.rs
git commit -m "feat: add From<Utf8Error> impl for XGBError"
```

---

## Phase 2: Build System Rewrite

### Task 2.1: Update xgboost-sys Cargo.toml with new build dependencies

**Files:**
- Modify: `xgboost-sys/Cargo.toml`

- [ ] **Step 1: Replace build-dependencies**

Replace lines 19-30 in `xgboost-sys/Cargo.toml`:

```toml
[build-dependencies]
bindgen = { version = "0.71" }
ureq = "2.9"
zip = "2.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
```

Remove: `dunce`, `reqwest`, `cmake` from build-dependencies. Remove `reqwest` from `[features]`.

- [ ] **Step 2: Remove old features, simplify**

Replace the features section (lines 25-29):

```toml
[features]
default = []
```

- [ ] **Step 3: Remove the `exclude` field**

Delete lines 12-14:
```toml
exclude = [
    "lib/*",
]
```

- [ ] **Step 4: Commit**

```bash
git add xgboost-sys/Cargo.toml
git commit -m "build: update xgboost-sys build dependencies for PyPI-based download"
```

---

### Task 2.2: Rewrite xgboost-sys/build.rs

**Files:**
- Overwrite: `xgboost-sys/build.rs`

- [ ] **Step 1: Write the complete new build.rs**

Replace the entire content of `xgboost-sys/build.rs` with a new implementation that:
1. Guards against docs.rs (checks `DOCS_RS` and `CARGO_CFG_DOCSRS` env vars → early return)
2. Fetches XGBoost version from `XGBOOST_VERSION` env var (default `"3.0.5"`)
3. Queries `https://pypi.org/pypi/xgboost/{version}/json` to find matching wheel URL by platform keyword
4. Downloads the wheel into memory and extracts `libxgboost.so`/`libxgboost.dylib`/`xgboost.dll` via zip
5. Downloads `c_api.h` from GitHub raw by version tag, with SHA256 verification
6. Runs bindgen on the header, writes `bindings.rs` to `OUT_DIR`
7. Sets rpath for macOS (`@loader_path`) and Linux (`$ORIGIN`)
8. Copies the shared library to the target directory
9. Emits `cargo:rerun-if-changed=build.rs` and `cargo:rerun-if-env-changed=XGBOOST_VERSION`

```rust
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct PyPIResponse {
    urls: Vec<PyPIUrl>,
}

#[derive(Deserialize)]
struct PyPIUrl {
    url: String,
    filename: String,
}

fn xgboost_version() -> String {
    env::var("XGBOOST_VERSION").unwrap_or_else(|_| "3.2.0".to_string())
}

fn get_platform_keyword() -> (&'static str, &'static str) {
    let target = env::var("TARGET").unwrap();
    let os = if target.contains("apple-darwin") {
        "macos"
    } else if target.contains("linux") {
        "linux"
    } else if target.contains("windows") {
        "windows"
    } else {
        panic!("Unsupported target OS: {}", target);
    };
    let keyword = match os {
        "linux" => "manylinux",
        "macos" => "macosx",
        "windows" => "win_amd64",
        _ => unreachable!(),
    };
    (os, keyword)
}

fn lib_filename(os: &str) -> &'static str {
    match os {
        "windows" => "xgboost.dll",
        "macos" => "libxgboost.dylib",
        _ => "libxgboost.so",
    }
}

fn download_with_retries(url: &str, max_retries: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut last_err = None;
    for attempt in 0..max_retries {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(200 * 2u64.pow(attempt - 1)));
        }
        match ureq::get(url).call() {
            Ok(resp) => {
                let mut buf = Vec::new();
                resp.into_reader().read_to_end(&mut buf)?;
                return Ok(buf);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(format!("download failed after {} retries: {:?}", max_retries, last_err).into())
}

fn sha256_hex(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

/// Known SHA256 checksums for c_api.h by XGBoost version.
fn known_header_checksum(version: &str) -> &'static str {
    match version {
        "3.2.0" => "30dd7487d154c84b7ba451bbc4f67637d7e338e64e9049b890468d7145f8508e",
        _ => panic!("No known SHA256 checksum for XGBoost version {}. Add it to known_header_checksum().", version),
    }
}

fn download_header(out_dir: &Path, version: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let include_dir = out_dir.join("include").join("xgboost");
    fs::create_dir_all(&include_dir)?;

    let url = format!(
        "https://raw.githubusercontent.com/dmlc/xgboost/v{}/include/xgboost/c_api.h",
        version
    );
    let data = download_with_retries(&url, 3)?;

    let expected = known_header_checksum(version);
    let actual = sha256_hex(&data);
    if actual != expected {
        return Err(format!(
            "SHA256 mismatch for c_api.h v{}: expected {}, got {}",
            version, expected, actual
        ).into());
    }
    println!("cargo:warning=Verified SHA256 for c_api.h v{}", version);

    let path = include_dir.join("c_api.h");
    fs::write(&path, &data)?;
    Ok(include_dir)
}

fn download_wheel(out_dir: &Path, version: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let lib_dir = out_dir.join("libs");
    let (os, keyword) = get_platform_keyword();
    let lib_name = lib_filename(os);
    let lib_path = lib_dir.join(lib_name);

    if lib_path.exists() {
        println!("cargo:warning=Using cached library: {}", lib_path.display());
        return Ok(lib_dir);
    }

    fs::create_dir_all(&lib_dir)?;

    // Query PyPI JSON API
    let pypi_url = format!("https://pypi.org/pypi/xgboost/{}/json", version);
    let resp = ureq::get(&pypi_url).call()?;
    let pypi: PyPIResponse = resp.into_json()?;

    let wheel = pypi.urls.iter()
        .find(|u| u.filename.ends_with(".whl") && u.filename.contains(keyword))
        .ok_or_else(|| format!("No matching wheel found for keyword '{}' in version {}", keyword, version))?;

    println!("cargo:warning=Downloading wheel: {}", wheel.filename);
    let wheel_bytes = download_with_retries(&wheel.url, 3)?;

    // Extract shared library from wheel (wheel is a ZIP file)
    let cursor = io::Cursor::new(wheel_bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let mut found = false;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        if name.ends_with(lib_name) {
            let mut dest = File::create(&lib_path)?;
            io::copy(&mut file, &mut dest)?;
            found = true;
            println!("cargo:warning=Extracted {} from wheel", lib_name);
            break;
        }
    }
    if !found {
        return Err(format!("{} not found in wheel", lib_name).into());
    }

    Ok(lib_dir)
}

fn set_rpath(lib_dir: &Path, os: &str) {
    if os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    } else if os == "linux" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}

fn main() {
    // docs.rs guard: skip all network operations
    if env::var("DOCS_RS").is_ok() || env::var("CARGO_CFG_DOCSRS").is_ok() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let version = xgboost_version();
    let (os, _) = get_platform_keyword();

    // Download header + verify SHA256
    let include_dir = download_header(&out_dir, &version)
        .expect("Failed to download XGBoost headers");

    // Download wheel + extract shared library
    let lib_dir = download_wheel(&out_dir, &version)
        .expect("Failed to download XGBoost wheel");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("c_api.h").to_string_lossy())
        .allowlist_function("XGB.*")
        .allowlist_function("XGD.*")
        .allowlist_type("BoosterHandle")
        .allowlist_type("DMatrixHandle")
        .allowlist_type("bst_ulong")
        .size_t_is_usize(true)
        .generate_comments(false)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings");

    // Set rpath and link
    set_rpath(&lib_dir, os);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=xgboost");

    // Only rerun when build.rs or XGBOOST_VERSION changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=XGBOOST_VERSION");
}
```

- [ ] **Step 2: Add docs.rs metadata to xgboost-sys/Cargo.toml**

Add to end of `xgboost-sys/Cargo.toml`:

```toml
[package.metadata.docs.rs]
rustdoc-args = ["--cfg", "docsrs"]
```

- [ ] **Step 3: Build to verify the new build system works**

Run: `cargo clean && cargo build --verbose`
Expected: Downloads headers from GitHub, downloads wheel from PyPI, extracts library, generates bindings, builds successfully.

- [ ] **Step 4: Run tests**

Run: `cargo test --verbose`
Expected: All tests pass with the new shared library.

- [ ] **Step 5: Commit**

```bash
git add xgboost-sys/build.rs xgboost-sys/Cargo.toml
git commit -m "build: rewrite build.rs to download headers from GitHub and lib from PyPI wheel"
```

---

### Task 2.3: Delete prebuilt libraries and old config

**Files:**
- Delete: `xgboost-sys/lib/` directory
- Delete: `xgboost-sys/.cargo/config.toml`
- Read then delete: `.travis.yml`
- Keep: `xgboost-sys/lib/README.md` (if it contains useful docs)

- [ ] **Step 1: Check what's in the lib README**

Run: `cat xgboost-sys/lib/README.md`

- [ ] **Step 2: Remove old prebuilt libs**

```bash
git rm -r xgboost-sys/lib/
```

- [ ] **Step 3: Remove old cargo config**

```bash
git rm xgboost-sys/.cargo/config.toml
```

- [ ] **Step 4: Delete .travis.yml**

```bash
git rm .travis.yml
```

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove prebuilt libraries, old cargo config, and travis config"
```

---


## Phase 3: API Simplification — Flat String Key-Value

The core change: delete all parameter structs. Parameters are passed as `&[(&str, &str)]` key-value pairs, exactly like Python's `xgb.train(params_dict, dtrain)`.

### Task 3.1: Rewrite Booster::train and Booster::new for flat string params

**Files:**
- Modify: `src/booster.rs`

- [ ] **Step 1: Replace train() signature and implementation**

Replace `Booster::train` (lines 189-244) with a new implementation that takes `&[(&str, &str)]` instead of `&TrainingParameters`:

```rust
    /// Train a new Booster model.
    ///
    /// * `params` - XGBoost parameters as key-value string pairs
    /// * `dtrain` - training data matrix
    /// * `boost_rounds` - number of boosting iterations
    /// * `eval_sets` - optional evaluation datasets with names
    pub fn train(
        params: &[(&str, &str)],
        dtrain: &DMatrix,
        boost_rounds: u32,
        eval_sets: Option<&[(&DMatrix, &str)]>,
    ) -> XGBResult<Self> {
        let mut bst = Booster::new(params)?;

        for i in 0..boost_rounds as i32 {
            bst.update(dtrain, i)?;

            if let Some(eval_sets) = eval_sets {
                let dmat_eval_results = bst.eval_set(eval_sets, i)?;
                let mut eval_dmat_results = std::collections::BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(std::collections::BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }
                print!("[{}]", i);
                for (eval_name, dmat_results) in eval_dmat_results {
                    for (dmat_name, result) in dmat_results {
                        print!("\t{}-{}:{}", dmat_name, eval_name, result);
                    }
                }
                println!();
            }
        }

        Ok(bst)
    }
```

- [ ] **Step 2: Simplify Booster::new**

Replace `Booster::new` and `new_with_cached_dmats` (lines 100-120):

```rust
    /// Create a new Booster with given parameters.
    pub fn new(params: &[(&str, &str)]) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(
            ptr::null(),
            0,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        for (key, value) in params {
            booster.set_param(key, value)?;
        }
        Ok(booster)
    }
```

- [ ] **Step 3: Remove set_params method**

Remove `set_params` (lines 247-253) which takes the deleted `BoosterParameters` type. `set_param` (single key-value, lines 698-706) stays.

- [ ] **Step 4: Remove BoosterParameters/TrainingParameters imports and CustomObjective**

Replace line 13:
```rust
use crate::parameters::{BoosterParameters, TrainingParameters};
```
with: *(remove the line entirely)*

Remove line 15:
```rust
pub type CustomObjective = fn(&[f32], &DMatrix) -> (Vec<f32>, Vec<f32>);
```

- [ ] **Step 5: Build to verify**

Run: `cargo build --verbose`
Expected: Compile errors referencing deleted types. Expected — fixed in next tasks.

- [ ] **Step 6: Commit**

```bash
git add src/booster.rs
git commit -m "refactor: rewrite Booster::train/new with flat &[(&str, &str)] params"
```

---

### Task 3.2: Delete parameters module and remove dead dependencies

**Files:**
- Delete: `src/parameters/` directory (all files)
- Modify: `src/lib.rs`
- Modify: `Cargo.toml`
- Modify: `src/booster.rs`

- [ ] **Step 1: Delete parameters directory**

```bash
git rm -r src/parameters/
```

- [ ] **Step 2: Update lib.rs**

Remove from `src/lib.rs`:
```rust
#[macro_use]
extern crate derive_builder;
```
```rust
extern crate indexmap;
```
```rust
pub mod parameters;
```

Replace the old doc example (lines 11-57) with:

```rust
//! # Basic usage example
//!
//! ```
//! use xgb::{DMatrix, Booster};
//!
//! fn main() {
//!     let x_train = &[1.0, 1.0, 1.0,
//!                     1.0, 1.0, 0.0,
//!                     1.0, 1.0, 1.0,
//!                     0.0, 0.0, 0.0,
//!                     1.0, 1.0, 1.0];
//!     let mut dtrain = DMatrix::from_dense(x_train, 5).unwrap();
//!     dtrain.set_labels(&[1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
//!
//!     let x_test = &[0.7, 0.9, 0.6];
//!     let mut dtest = DMatrix::from_dense(x_test, 1).unwrap();
//!     dtest.set_labels(&[1.0]).unwrap();
//!
//!     let eval_sets = &[(&dtrain, "train"), (&dtest, "test")];
//!
//!     let bst = Booster::train(
//!         &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
//!         &dtrain,
//!         2,
//!         Some(eval_sets),
//!     ).unwrap();
//!
//!     println!("{:?}", bst.predict(&dtest).unwrap());
//! }
//! ```
```

- [ ] **Step 3: Remove dead dependencies from Cargo.toml**

In `Cargo.toml` `[dependencies]`, remove:
- `derive_builder = "0.20"`
- `indexmap = "2.7"`

Final `[dependencies]`:
```toml
[dependencies]
xgboost-sys = { package = "xgboost_lib-sys", path = "xgboost-sys", version = "0.1.0" }
libc = "0.2"
log = "0.4"
tempfile = "3.15"
```

- [ ] **Step 4: Replace IndexMap with HashMap in booster.rs**

In `booster.rs`, remove the `indexmap` import (line 10) and replace all `IndexMap` references with `HashMap`. Specifically:
- Remove `use indexmap::IndexMap;`
- `eval_set` return type: `HashMap<String, HashMap<String, f32>>`
- `parse_eval_string` signature: same change

- [ ] **Step 5: Build to verify**

Run: `cargo build --verbose`
Expected: Compiles (test failures expected — fixed next).

- [ ] **Step 6: Commit**

```bash
git add src/ Cargo.toml
git commit -m "refactor: delete parameters module, remove derive_builder and indexmap deps"
```

---

### Task 3.3: Rewrite tests for flat-params API

**Files:**
- Modify: `src/booster.rs` (test sections)

- [ ] **Step 1: Rewrite load_test_booster helper**

```rust
    fn load_test_booster() -> Booster {
        let dmat = read_train_matrix().expect("Reading train matrix failed");
        Booster::new(&[]).expect("Creating Booster failed")
    }
```

- [ ] **Step 2: Rewrite save_and_load_from_buffer test**

Replace `BoosterParameters::default()` with `&[]`:
```rust
    let mut booster = Booster::new(&[]).unwrap();
```

- [ ] **Step 3: Rewrite predict test**

Replace entire predict test block:

```rust
    #[test]
    fn predict() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

        let params = &[
            ("max_depth", "2"),
            ("eta", "1.0"),
            ("objective", "binary:logistic"),
            ("eval_metric", "map@4-,logloss,error@0.5"),
        ];

        let mut booster = Booster::new(params).unwrap();
        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let train_metrics = booster.evaluate(&dmat_train).unwrap();
        let logloss = train_metrics.get("logloss").expect("logloss metric missing");
        assert!((*logloss - 0.006634271).abs() < 1e-6, "train logloss was {}", logloss);

        let test_metrics = booster.evaluate(&dmat_test).unwrap();
        let test_logloss = test_metrics.get("logloss").expect("test logloss metric missing");
        assert!((*test_logloss - 0.0069199526).abs() < 1e-6, "test logloss was {}", test_logloss);

        let v = booster.predict(&dmat_test).unwrap();
        assert_eq!(v.len(), dmat_test.num_rows());

        let expected_start = [
            0.0050151693, 0.9884467, 0.0050151693, 0.0050151693,
            0.026636455, 0.11789363, 0.9884467, 0.01231471,
            0.9884467, 0.00013656063,
        ];
        let eps = 1e-6;
        for (pred, expected) in v.iter().zip(&expected_start) {
            assert!((pred - expected).abs() < eps, "pred={}, expected={}", pred, expected);
        }
    }
```

- [ ] **Step 4: Rewrite predict_matrix test**

Same pattern — flat params, manual update loop.

- [ ] **Step 5: Rewrite predict_leaf, predict_contributions, predict_interactions**

Each follows the exact same pattern:
```rust
    let params = &[
        ("max_depth", "2"),
        ("eta", "1.0"),
        ("objective", "binary:logistic"),
        ("eval_metric", "logloss"),
    ];
    let mut booster = Booster::new(params).unwrap();
    for i in 0..N { booster.update(&dmat_train, i).expect("update failed"); }
    // ... assertions unchanged ...
```

- [ ] **Step 6: Rewrite dump_model test**

```rust
    let params = &[
        ("max_depth", "2"),
        ("eta", "1.0"),
        ("objective", "binary:logistic"),
    ];
    let mut booster = Booster::new(params).unwrap();
    for i in 0..10 {
        booster.update(&dmat_train, i).expect("update failed");
    }
    // ... dump assertions unchanged ...
```

- [ ] **Step 7: Remove parse_eval_string test**

`parse_eval_string` is now a private helper, no direct unit test needed. Delete lines 1227-1242.

- [ ] **Step 8: Run all tests**

Run: `cargo test --verbose`
Expected: All tests pass. If metric name format differs from old assertions, adjust test keys.

- [ ] **Step 9: Commit**

```bash
git add src/booster.rs
git commit -m "test: rewrite all tests for flat string key-value param API"
```

---

### Task 3.4: Update examples for flat-params API

**Files:**
- Overwrite: `examples/basic/src/main.rs`
- Delete: `examples/custom_objective/`
- Overwrite: `examples/generalised_linear_model/src/main.rs`
- Overwrite: `examples/multiclass_classification/src/main.rs`

- [ ] **Step 1: Rewrite examples/basic/src/main.rs**

```rust
use xgb::{DMatrix, Booster};

fn main() {
    env_logger::init();

    let dtrain = DMatrix::load(r#"{"uri": "../../xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
    let dtest = DMatrix::load(r#"{"uri": "../../xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();
    println!("Train: {}x{}, Test: {}x{}", dtrain.num_rows(), dtrain.num_cols(), dtest.num_rows(), dtest.num_cols());

    let eval_sets = &[(&dtest, "test"), (&dtrain, "train")];
    let bst = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dtrain,
        2,
        Some(eval_sets),
    ).unwrap();

    let preds = bst.predict(&dtest).unwrap();
    let labels = dtest.get_labels().unwrap();
    let num_correct: usize = preds.iter().zip(labels.iter())
        .filter(|(p, l)| (*p > 0.5) as u8 as f32 == *l)
        .count();
    println!("error={}", 1.0 - num_correct as f32 / preds.len() as f32);

    bst.save("xgb.json").unwrap();
    let bst2 = Booster::load("xgb.json").unwrap();
    assert_eq!(bst.predict(&dtest).unwrap(), bst2.predict(&dtest).unwrap());
}
```

- [ ] **Step 2: Delete custom_objective example**

```bash
git rm -r examples/custom_objective/
```

Update `examples/runall.sh` and `examples/README.md` to remove references.

- [ ] **Step 3: Rewrite examples/generalised_linear_model/src/main.rs**

```rust
use xgb::{DMatrix, Booster};

fn main() {
    env_logger::init();

    let dtrain = DMatrix::load(r#"{"uri": "../../xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
    let dtest = DMatrix::load(r#"{"uri": "../../xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

    let eval_sets = &[(&dtest, "test"), (&dtrain, "train")];
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
    let num_errors: usize = preds.iter().zip(labels.iter())
        .filter(|(p, l)| (*p > 0.5) as u8 as f32 != *l)
        .count();
    println!("error={}", num_errors as f32 / preds.len() as f32);
}
```

- [ ] **Step 4: Rewrite examples/multiclass_classification/src/main.rs** (keep data loading helpers)

```rust
use std::io::{BufRead, BufReader};
use std::fs::File;
use xgb::{DMatrix, Booster};

fn main() {
    env_logger::init();

    // ... download_dataset and load_train_test_dmats functions unchanged ...

    let (dtrain, dtest) = load_train_test_dmats("dermatology.data");
    let eval_sets = &[(&dtrain, "train"), (&dtest, "test")];

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
```

- [ ] **Step 5: Build examples**

Run: `cargo build --examples --verbose`
Expected: All examples compile.

- [ ] **Step 6: Commit**

```bash
git add examples/
git commit -m "refactor: rewrite examples for flat string key-value API, remove custom_objective"
```

---

### Task 3.5: Update README and docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README Basic usage example**

Replace lines 31-93 with:

```rust
use xgb::{DMatrix, Booster};

fn main() {
    let x_train = &[1.0, 1.0, 1.0,
                    1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0];
    let mut dtrain = DMatrix::from_dense(x_train, 5).unwrap();
    dtrain.set_labels(&[1.0, 1.0, 1.0, 0.0, 1.0]).unwrap();

    let x_test = &[0.7, 0.9, 0.6];
    let mut dtest = DMatrix::from_dense(x_test, 1).unwrap();
    dtest.set_labels(&[1.0]).unwrap();

    let eval_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // All XGBoost parameters as string key-value pairs
    let bst = Booster::train(
        &[("max_depth", "2"), ("eta", "1.0"), ("objective", "binary:logistic")],
        &dtrain,
        2,
        Some(eval_sets),
    ).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
}
```

- [ ] **Step 2: Update Requirements section**

Remove references to `libomp`, `libclang-dev` — PyPI wheel is self-contained.

- [ ] **Step 3: Simplify "Use prebuilt xgboost" section**

Replace with a short note about `XGBOOST_VERSION` env var.

- [ ] **Step 4: Build docs**

Run: `cargo doc --no-deps`
Expected: Docs build cleanly.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README for flat string key-value API"
```

## Phase 4: Engineering Polish

### Task 4.1: Reset version to 0.1.0 and add workspace

**Files:**
- Modify: `Cargo.toml` (root)
- Modify: `xgboost-sys/Cargo.toml`

- [ ] **Step 1: Add workspace to root Cargo.toml**

Add after `[package]` section in root `Cargo.toml`:

```toml
[workspace]
members = [".", "xgboost-sys"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/marcomq/rust-xgboost"
```

- [ ] **Step 2: Update root package version**

Change line 3: `version = "3.0.5"` to `version = "0.1.0"`

- [ ] **Step 3: Update root package to use workspace fields**

```toml
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
```

- [ ] **Step 4: Update xgboost-sys Cargo.toml version**

Change `xgboost-sys/Cargo.toml` line 3: `version = "3.0.4"` to `version = "0.1.0"`

- [ ] **Step 5: Fix the documentation URL**

In root `Cargo.toml` line 9, change:
```toml
documentation = "https://docs.rs/xgboost_lib"
```
to:
```toml
documentation = "https://docs.rs/xgb"
```

- [ ] **Step 6: Add docs.rs metadata to root Cargo.toml**

```toml
[package.metadata.docs.rs]
rustdoc-args = ["--cfg", "docsrs"]
```

- [ ] **Step 7: Update xgboost-sys dependency reference**

In root `Cargo.toml` line 14, change version to `"0.1.0"`:
```toml
xgboost-sys = { package = "xgboost_lib-sys", path = "xgboost-sys", version = "0.1.0" }
```

- [ ] **Step 8: Clean up features section**

Remove `local_build`, `cuda`, `use_prebuilt_xgb` features — all replaced by PyPI wheel download:

```toml
[features]
default = []
```

- [ ] **Step 9: Build to verify**

Run: `cargo build --verbose`
Expected: Builds with version 0.1.0.

- [ ] **Step 10: Commit**

```bash
git add Cargo.toml xgboost-sys/Cargo.toml
git commit -m "chore: reset version to 0.1.0, add workspace, fix docs.rs metadata"
```

---

### Task 4.2: Rewrite CHANGELOG

**Files:**
- Overwrite: `CHANGELOG.md`

- [ ] **Step 1: Write new CHANGELOG**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- Rust bindings for XGBoost 3.2.x C API
- `DMatrix` for loading and manipulating data matrices (dense, CSR, CSC, LibSVM, binary)
- `Booster` for training, prediction, evaluation, model save/load
- Flat string key-value parameter API (like Python's `xgb.train()`)
- Shared library auto-download from PyPI wheels via build.rs
- Cross-platform support: Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64)

[0.1.0]: https://github.com/marcomq/rust-xgboost/releases/tag/v0.1.0
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: rewrite CHANGELOG for 0.1.0"
```

---

### Task 4.3: Consolidate CI workflows

**Files:**
- Modify: `.github/workflows/linux.yml`
- Modify: `.github/workflows/macos.yml`
- Modify: `.github/workflows/windows.yml`
- Delete: `.github/workflows/linux_arm64.yml`

- [ ] **Step 1: Update Linux CI with full checks**

Replace `.github/workflows/linux.yml`:

```yaml
name: Linux

on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always

jobs:
  linux:
    name: linux (x86_64)
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Build
      run: cargo build --verbose

    - name: Run tests
      run: cargo test --verbose

    - name: Clippy
      run: cargo clippy -- -D warnings

    - name: Format check
      run: cargo fmt --check

    - name: Doc
      run: cargo doc --no-deps --document-private-items

  linux-arm64:
    name: linux (aarch64)
    runs-on: ubuntu-22.04-arm

    steps:
    - uses: actions/checkout@v4

    - name: Build
      run: cargo build --verbose

    - name: Run tests
      run: cargo test --verbose
```

- [ ] **Step 2: Simplify macOS CI**

Replace `.github/workflows/macos.yml` (remove `brew install libomp`):

```yaml
name: macOS

on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always

jobs:
  macos:
    name: macos
    runs-on: macos-latest

    steps:
    - uses: actions/checkout@v4

    - name: Build
      run: cargo build --verbose

    - name: Run tests
      run: cargo test --verbose
```

- [ ] **Step 3: Simplify Windows CI**

Replace `.github/workflows/windows.yml` (remove `submodules: recursive`):

```yaml
name: Windows

on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always

jobs:
  windows:
    runs-on: windows-latest

    steps:
    - uses: actions/checkout@v4

    - name: Build
      run: cargo build --verbose

    - name: Run tests
      run: cargo test --verbose
```

- [ ] **Step 4: Delete linux_arm64 workflow**

```bash
git rm .github/workflows/linux_arm64.yml
```

- [ ] **Step 5: Commit**

```bash
git add .github/
git commit -m "ci: consolidate workflows, add clippy/fmt/doc, remove brew and submodule deps"
```

---

### Task 4.4: Final verification

- [ ] **Step 1: Full build**

Run: `cargo build --verbose`
Expected: Clean build, no warnings.

- [ ] **Step 2: Full test suite**

Run: `cargo test --verbose`
Expected: All tests pass.

- [ ] **Step 3: Clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings.

- [ ] **Step 4: Format check**

Run: `cargo fmt --check`
Expected: No formatting issues.

- [ ] **Step 5: Doc build**

Run: `cargo doc --no-deps`
Expected: Docs build successfully.

- [ ] **Step 6: Final commit**

```bash
git add -A
git status
git commit -m "chore: final verification pass"
```
