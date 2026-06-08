use libc::{c_float, c_uint};
use std::{ffi, path::Path, ptr, slice};

use super::{XGBError, XGBResult, NAN_SENTINEL};

// XGBoost-defined field names for DMatrix metadata (see C API XGDMatrixSetFloatInfo / SetUIntInfo).
//
// | type   | field              | meaning                            |
// |--------|--------------------|------------------------------------|
// | float  | label              | ground truth labels                |
// | float  | weight             | instance weights                   |
// | float  | base_margin        | base prediction before boosting    |
// | float  | label_lower_bound  | lower bound (censored regression)  |
// | float  | label_upper_bound  | upper bound (censored regression)  |
// | float  | feature_weights    | per-feature column weights         |
// | uint   | group_ptr          | cumulative group offsets (ranking) |
// | uint   | group              | group sizes (ranking)              |
// | uint   | qid                | query ID (ranking)                 |
pub static KEY_GROUP_PTR: &str = "group_ptr";
pub static KEY_GROUP: &str = "group";
pub static KEY_LABEL: &str = "label";
pub static KEY_WEIGHT: &str = "weight";
pub static KEY_BASE_MARGIN: &str = "base_margin";
pub static KEY_LABEL_LOWER_BOUND: &str = "label_lower_bound";
pub static KEY_LABEL_UPPER_BOUND: &str = "label_upper_bound";
pub static KEY_QID: &str = "qid";

/// Data matrix used throughout XGBoost for training/predicting [`Booster`](struct.Booster.html) models.
///
/// It's used as a container for both features (i.e. a row for every instance), and an optional true label for that
/// instance (as an `f32` value).
///
/// Can be created files, or from dense or sparse
/// ([CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
/// or [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrices.
///
/// # Examples
///
/// ## Load from file
///
/// Load matrix from file in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) or binary format.
///
/// ```should_panic
/// use xgb::DMatrix;
///
/// let dmat = DMatrix::load(r#"{"uri": "somefile.txt?format=csv"}"#).unwrap();
/// ```
///
/// ## Create from dense array
///
/// ```
/// use xgb::DMatrix;
///
/// let data = &[1.0, 0.5, 0.2, 0.2,
///              0.7, 1.0, 0.1, 0.1,
///              0.2, 0.0, 0.0, 1.0];
/// let num_rows = 3;
/// let mut dmat = DMatrix::from_dense(data, num_rows).unwrap();
/// assert_eq!(dmat.shape(), (3, 4));
///
/// // set true labels for each row
/// dmat.set_label(&[1.0, 0.0, 1.0]);
/// ```
///
/// ## Create from sparse CSR matrix
///
/// Create from sparse representation of
/// ```text
/// [[1.0, 0.0, 2.0],
///  [0.0, 0.0, 3.0],
///  [4.0, 5.0, 6.0]]
/// ```
///
/// ```
/// use xgb::DMatrix;
///
/// let indptr = &[0, 1, 2, 6];
/// let indices = &[0, 2, 2, 0, 1, 2];
/// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let dmat = DMatrix::from_csc(indptr, indices, data, None).unwrap();
/// assert_eq!(dmat.shape(), (3, 3));
/// ```
#[derive(Debug)]
pub struct DMatrix {
    pub(super) handle: xgboost_sys::DMatrixHandle,
    num_rows: usize,
    num_cols: usize,
}

impl DMatrix {
    /// Construct a new instance from a DMatrixHandle created by the XGBoost C API.
    fn new(handle: xgboost_sys::DMatrixHandle) -> XGBResult<Self> {
        // number of rows/cols are frequently read throughout applications, so more convenient to pull them out once
        // when the matrix is created, instead of having to check errors each time XGDMatrixNum* is called
        let mut out = 0;
        xgb_call!(xgboost_sys::XGDMatrixNumRow(handle, &mut out))?;
        let num_rows = out as usize;

        let mut out = 0;
        xgb_call!(xgboost_sys::XGDMatrixNumCol(handle, &mut out))?;
        let num_cols = out as usize;

        trace!("Loaded DMatrix with shape: {}x{}", num_rows, num_cols);
        Ok(DMatrix {
            handle,
            num_rows,
            num_cols,
        })
    }

    /// Return the native-endian typestr for the given float width (4 = f32, 8 = f64).
    fn ftypestr(width: u8) -> String {
        let prefix = if cfg!(target_endian = "big") { ">" } else { "<" };
        format!("{}f{}", prefix, width)
    }

    /// Return the native-endian typestr for the given unsigned integer width.
    fn utypestr(width: u8) -> String {
        let prefix = if cfg!(target_endian = "big") { ">" } else { "<" };
        format!("{}u{}", prefix, width)
    }

    /// Create a new `DMatrix` from dense array in row-major order.
    ///
    /// E.g. the matrix
    /// ```text
    /// [[1.0, 2.0],
    ///  [3.0, 4.0],
    ///  [5.0, 6.0]]
    /// ```
    /// would be represented converted into a `DMatrix` with
    /// ```
    /// use xgb::DMatrix;
    ///
    /// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let num_rows = 3;
    /// let dmat = DMatrix::from_dense(data, num_rows).unwrap();
    /// ```
    pub fn from_dense(data: &[f32], num_rows: usize) -> XGBResult<Self> {
        let num_cols = data.len() / num_rows;
        let mut handle = ptr::null_mut();
        let array_json = json_str!(
            "data" => serde_json::json!([data.as_ptr() as usize, false]),
            "shape" => [num_rows, num_cols],
            "typestr" => Self::ftypestr(4),
            "version" => 3,
        );
        let config = json_str!("missing" => NAN_SENTINEL, "nthread" => 0, "data_split_mode" => 0);
        xgb_call!(xgboost_sys::XGDMatrixCreateFromDense(
            array_json.as_ptr(),
            config.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) matrix.
    pub fn from_csr(indptr: &[usize], indices: &[usize], data: &[f32], num_cols: Option<usize>) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let indptr: Vec<u64> = indptr.iter().map(|x| *x as u64).collect();
        let indices: Vec<u32> = indices.iter().map(|x| *x as u32).collect();
        let ncol = num_cols.unwrap_or(0) as u64;
        let indptr_json = json_str!("data" => serde_json::json!([indptr.as_ptr() as usize, false]), "shape" => [indptr.len()], "typestr" => Self::utypestr(8), "version" => 3);
        let indices_json = json_str!("data" => serde_json::json!([indices.as_ptr() as usize, false]), "shape" => [indices.len()], "typestr" => Self::utypestr(4), "version" => 3);
        let data_json = json_str!("data" => serde_json::json!([data.as_ptr() as usize, false]), "shape" => [data.len()], "typestr" => Self::ftypestr(4), "version" => 3);
        let config = json_str!("missing" => NAN_SENTINEL, "nthread" => 0, "data_split_mode" => 0);
        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            ncol as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrix.
    pub fn from_csc(indptr: &[usize], indices: &[usize], data: &[f32], num_rows: Option<usize>) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let indptr: Vec<u64> = indptr.iter().map(|x| *x as u64).collect();
        let indices: Vec<u32> = indices.iter().map(|x| *x as u32).collect();
        let nrow = num_rows.unwrap_or(0) as u64;
        let indptr_json = json_str!("data" => serde_json::json!([indptr.as_ptr() as usize, false]), "shape" => [indptr.len()], "typestr" => Self::utypestr(8), "version" => 3);
        let indices_json = json_str!("data" => serde_json::json!([indices.as_ptr() as usize, false]), "shape" => [indices.len()], "typestr" => Self::utypestr(4), "version" => 3);
        let data_json = json_str!("data" => serde_json::json!([data.as_ptr() as usize, false]), "shape" => [data.len()], "typestr" => Self::ftypestr(4), "version" => 3);
        let config = json_str!("missing" => NAN_SENTINEL, "nthread" => 0, "data_split_mode" => 0);
        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSC(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            nrow as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from given file.
    ///
    /// Supports text files in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format, CSV,
    /// binary files written either by `save`, or from another XGBoost library.
    ///
    /// For more details on accepted formats, seem the
    /// [XGBoost input format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
    /// documentation.
    ///
    /// # LIBSVM format
    ///
    /// Specified data in a sparse format as:
    /// ```text
    /// <label> <index>:<value> [<index>:<value> ...]
    /// ```
    ///
    /// E.g.
    /// ```text
    /// 0 1:1 9:0 11:0
    /// 1 9:1 11:0.375 15:1
    /// 0 1:0 8:0.22 11:1
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading DMatrix from: {}", path.as_ref().display());
        let mut handle = ptr::null_mut();
        let config = json_str!("uri" => path.as_ref().to_string_lossy(), "silent" => 1);
        xgb_call!(xgboost_sys::XGDMatrixCreateFromURI(config.as_ptr(), &mut handle))?;
        DMatrix::new(handle)
    }

    /// Serialise this `DMatrix` as a binary file to given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing DMatrix to: {}", path.as_ref().display());
        let fname: ffi::CString = crate::path_to_c_str(path);
        let silent = true;
        xgb_call!(xgboost_sys::XGDMatrixSaveBinary(
            self.handle,
            fname.as_ptr(),
            silent as i32
        ))
    }

    /// Get the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Get the shape (rows x columns) of this matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }

    /// Get the number of non-missing values in the DMatrix.
    ///
    /// This is the count of elements that are not marked as missing in the data.
    pub fn num_nonmissing(&self) -> XGBResult<usize> {
        let mut out = 0u64;
        xgb_call!(xgboost_sys::XGDMatrixNumNonMissing(self.handle, &mut out))?;
        Ok(out as usize)
    }

    /// Get the data split mode (row-wise vs column-wise) for distributed computing.
    pub fn data_split_mode(&self) -> XGBResult<u64> {
        let mut out = 0u64;
        xgb_call!(xgboost_sys::XGDMatrixDataSplitMode(self.handle, &mut out))?;
        Ok(out)
    }

    /// Get a new DMatrix containing only the given row indices.
    ///
    /// If `allow_groups` is true, permits slicing a matrix that has group information set
    /// (used in learning-to-rank tasks).
    pub fn slice(&self, indices: &[usize], allow_groups: bool) -> XGBResult<DMatrix> {
        debug!("Slicing {} rows from DMatrix (allow_groups={})", indices.len(), allow_groups);
        let mut out_handle = ptr::null_mut();
        let indices: Vec<i32> = indices.iter().map(|x| *x as i32).collect();
        xgb_call!(xgboost_sys::XGDMatrixSliceDMatrixEx(
            self.handle,
            indices.as_ptr(),
            indices.len() as xgboost_sys::bst_ulong,
            &mut out_handle,
            allow_groups as i32
        ))?;
        DMatrix::new(out_handle)
    }

    /// Get float metadata by field name.
    ///
    /// Known fields: `"label"`, `"weight"`, `"base_margin"`,
    /// `"label_lower_bound"`, `"label_upper_bound"`.
    pub fn get_float_info(&self, field: &str) -> XGBResult<&[f32]> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_sys::XGDMatrixGetFloatInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        if out_len > 0 {
            Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_float, out_len as usize) })
        } else {
            Ok(&[0.0; 0])
        }
    }

    /// Set float metadata by field name.
    pub fn set_float_info(&mut self, field: &str, array: &[f32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixSetFloatInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }

    /// Set ground truth labels for each row.
    pub fn set_label(&mut self, labels: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_LABEL, labels)
    }

    /// Get ground truth labels.
    pub fn get_label(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_LABEL)
    }

    /// Set instance weights.
    pub fn set_weight(&mut self, weights: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_WEIGHT, weights)
    }

    /// Get instance weights.
    pub fn get_weight(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_WEIGHT)
    }

    /// Get unsigned integer metadata by field name.
    ///
    /// Known fields: `"group"`, `"group_ptr"`, `"qid"`.
    pub fn get_uint_info(&self, field: &str) -> XGBResult<&[u32]> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_sys::XGDMatrixGetUIntInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        if out_len > 0 {
            Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_uint, out_len as usize) })
        } else {
            Ok(&[0; 0])
        }
    }

    /// Set unsigned integer metadata by field name.
    pub fn set_uint_info(&mut self, field: &str, array: &[u32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixSetUIntInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }

    /// Get string array metadata by field name.
    ///
    /// Known fields: `"feature_name"`, `"feature_type"`.
    /// Returns `None` if no data has been set.
    pub fn get_str_feature_info(&self, field: &str) -> XGBResult<Option<Vec<String>>> {
        let field = ffi::CString::new(field).unwrap();
        let mut out_len = 0u64;
        let mut out_features: *mut *const std::os::raw::c_char = ptr::null_mut();
        xgb_call!(xgboost_sys::XGDMatrixGetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_features
        ))?;
        if out_len == 0 || out_features.is_null() {
            return Ok(None);
        }
        let len = out_len as usize;
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let c_str = unsafe { ffi::CStr::from_ptr(*out_features.add(i)) };
            result.push(c_str.to_string_lossy().into_owned());
        }
        Ok(Some(result))
    }

    /// Set string array metadata by field name.
    ///
    /// Pass an empty slice to reset.
    pub fn set_str_feature_info(&mut self, field: &str, features: &[String]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        let c_strings: Vec<ffi::CString> = features
            .iter()
            .map(|s| ffi::CString::new(s.as_str()).unwrap())
            .collect();
        let ptrs: Vec<*const std::os::raw::c_char> =
            c_strings.iter().map(|cs| cs.as_ptr()).collect();
        xgb_call!(xgboost_sys::XGDMatrixSetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            ptrs.as_ptr() as *mut *const std::os::raw::c_char,
            features.len() as u64
        ))
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        if let Err(e) = xgb_call!(xgboost_sys::XGDMatrixFree(self.handle)) {
            error!("XGDMatrixFree failed in drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read_train_matrix() -> XGBResult<DMatrix> {
        DMatrix::load("xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm")
    }

    #[test]
    fn read_matrix() {
        assert!(read_train_matrix().is_ok());
    }

    #[test]
    fn read_num_rows() {
        assert_eq!(read_train_matrix().unwrap().num_rows(), 6513);
    }

    #[test]
    fn read_num_cols() {
        assert_eq!(read_train_matrix().unwrap().num_cols(), 127);
    }

    #[test]
    fn writing_and_reading() {
        let dmat = read_train_matrix().unwrap();

        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let out_path = tmp_dir.path().join("dmat.bin");
        dmat.save(&out_path).unwrap();

        let dmat2 = DMatrix::load(out_path).unwrap();

        assert_eq!(dmat.num_rows(), dmat2.num_rows());
        assert_eq!(dmat.num_cols(), dmat2.num_cols());
        // TODO: check contents as well, if possible
    }

    #[test]
    fn get_set_labels() {
        let mut dmat = read_train_matrix().unwrap();
        let labels = dmat.get_label();
        assert!(labels.is_ok());
        let mut labels = labels.unwrap().to_vec();
        assert_eq!(labels.len(), 6513);

        labels[0] = 0.1;
        assert_ne!(dmat.get_label().unwrap(), labels);
        assert!(dmat.set_label(&labels).is_ok());
        assert_eq!(dmat.get_label().unwrap(), labels);
    }

    #[test]
    fn get_set_weights() {
        let mut dmat = read_train_matrix().unwrap();
        assert!(dmat.get_weight().unwrap().is_empty());

        let weight = [1.0, 10.0, 44.9555];
        assert!(dmat.set_weight(&weight).is_ok());
        assert_eq!(dmat.get_weight().unwrap(), weight);
    }

    #[test]
    fn get_set_base_margin() {
        let mut dmat = read_train_matrix().unwrap();
        let base_margin = dmat.get_float_info(KEY_BASE_MARGIN);
        assert!(base_margin.is_ok());
        assert!(base_margin.unwrap().is_empty());

        let base_margin = vec![0.00001; dmat.num_rows()];
        assert!(dmat.set_float_info(KEY_BASE_MARGIN, &base_margin).is_ok());
        assert_eq!(dmat.get_float_info(KEY_BASE_MARGIN).unwrap(), base_margin);
    }

    #[test]
    fn get_set_group() {
        let mut dmat = read_train_matrix().unwrap();
        assert!(dmat.get_uint_info(KEY_GROUP_PTR).unwrap().is_empty());

        let group = [1];
        assert!(dmat.set_uint_info(KEY_GROUP, &group).is_ok());
        assert_eq!(dmat.get_uint_info(KEY_GROUP_PTR).unwrap(), &[0, 1]);
    }

    #[test]
    fn from_csr() {
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 0); // https://github.com/dmlc/xgboost/pull/7265

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 10);
    }

    #[test]
    fn from_csc() {
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 4);

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 10);
        assert_eq!(dmat.num_cols(), 4);
    }

    #[test]
    fn from_dense() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_rows = 2;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(dmat.num_cols(), 3);

        let data = vec![1.0, 2.0, 3.0];
        let num_rows = 3;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 1);
    }

    #[test]
    fn slice_from_indices() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 2));

        assert_eq!(dmat.slice(&[], false).unwrap().shape(), (0, 2));
        assert_eq!(dmat.slice(&[1], false).unwrap().shape(), (1, 2));
        assert_eq!(dmat.slice(&[0, 1], false).unwrap().shape(), (2, 2));
        assert_eq!(dmat.slice(&[3, 2, 1], false).unwrap().shape(), (3, 2));
        // assert_eq!(dmat.slice(&[10, 11, 12], false).unwrap().shape(), (3, 2));
    }

    #[test]
    fn slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 3));

        assert_eq!(dmat.slice(&[0, 1, 2, 3], false).unwrap().shape(), (4, 3));
        assert_eq!(dmat.slice(&[0, 1], false).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[1, 0], false).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[0, 1, 2], false).unwrap().shape(), (3, 3));
        assert_eq!(dmat.slice(&[3, 2, 1], false).unwrap().shape(), (3, 3));
    }
}
