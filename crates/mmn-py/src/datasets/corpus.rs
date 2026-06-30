use mmn_data::{CorpusBatchSize, DatasetCorpus, DatasetCorpusConfig};
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyclass(name = "DatasetCorpus")]
pub struct PyDatasetCorpus {
    pub(crate) inner: DatasetCorpus,
}

#[pymethods]
impl PyDatasetCorpus {
    #[new]
    #[pyo3(signature = (use_two_files=true, rowfile=None, txtfile=None, sort_rows_by_complexity=true, rows_with_corpus_chunk="text", batch_size="row"))]
    pub fn new(
        use_two_files: bool,
        rowfile: Option<String>,
        txtfile: Option<String>,
        sort_rows_by_complexity: bool,
        rows_with_corpus_chunk: &str,
        batch_size: &str,
    ) -> PyResult<Self> {
        let bs = if batch_size == "row" {
            CorpusBatchSize::PerRow
        } else {
            CorpusBatchSize::Fixed(batch_size.parse().unwrap_or(24))
        };
        let inner = DatasetCorpus::load(DatasetCorpusConfig {
            use_two_files,
            rowfile,
            txtfile,
            sort_rows_by_complexity,
            rows_with_corpus_chunk: rows_with_corpus_chunk.to_string(),
            batch_size: bs,
        })
        .map_err(mmn_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.meta.rows
    }

    #[getter]
    fn format(&self) -> String {
        self.inner.meta.format.clone()
    }

    #[getter]
    fn type_(&self) -> String {
        "corpus".into()
    }

    /// `"row"` for per-row batches, otherwise the fixed batch size as a decimal string.
    #[getter]
    fn corpus_batch_size(&self) -> String {
        match self.inner.batch_size {
            CorpusBatchSize::PerRow => "row".into(),
            CorpusBatchSize::Fixed(n) => n.to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetCorpus(rows={}, format={:?}, type='corpus')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}
