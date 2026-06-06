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
        _ => panic!(
            "No known SHA256 checksum for XGBoost version {}. Add it to known_header_checksum().",
            version
        ),
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
        )
        .into());
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

    let wheel = pypi
        .urls
        .iter()
        .find(|u| u.filename.ends_with(".whl") && u.filename.contains(keyword))
        .ok_or_else(|| {
            format!(
                "No matching wheel found for keyword '{}' in version {}",
                keyword, version
            )
        })?;

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
    let include_dir =
        download_header(&out_dir, &version).expect("Failed to download XGBoost headers");

    // Download wheel + extract shared library
    let lib_dir =
        download_wheel(&out_dir, &version).expect("Failed to download XGBoost wheel");

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
