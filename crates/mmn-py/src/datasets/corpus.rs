use mmn_data::{CorpusBatchSize, CorpusRow, DatasetCorpus, DatasetCorpusConfig, DatasetMeta, DatasetType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

/// Plain-text corpus for next-token language modeling, loaded from files or an
/// in-memory list of strings (`data=["chunk one", "chunk two"]`).
#[pyclass(name = "DatasetCorpus")]
pub struct PyDatasetCorpus {
    pub(crate) inner: DatasetCorpus,
}

fn corpus_from_memory(chunks: Vec<String>, sort_rows_by_complexity: bool, bs: CorpusBatchSize) -> DatasetCorpus {
    let mut rows: Vec<CorpusRow> = chunks
        .into_iter()
        .map(|text| {
            let complexity = text.len() as f32;
            CorpusRow { text, complexity }
        })
        .collect();
    if sort_rows_by_complexity {
        rows.sort_by(|a, b| {
            a.complexity
                .partial_cmp(&b.complexity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    DatasetCorpus {
        meta: DatasetMeta {
            rows: rows.len(),
            format: "memory".into(),
            dataset_type: DatasetType::Corpus,
        },
        rows,
        batch_size: bs,
    }
}

#[pymethods]
impl PyDatasetCorpus {
    #[new]
    #[pyo3(signature = (use_two_files=true, rowfile=None, txtfile=None, sort_rows_by_complexity=true, rows_with_corpus_chunk="text", batch_size="row", data=None))]
    pub fn new(
        use_two_files: bool,
        rowfile: Option<String>,
        txtfile: Option<String>,
        sort_rows_by_complexity: bool,
        rows_with_corpus_chunk: &str,
        batch_size: &str,
        data: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let bs = if batch_size == "row" {
            CorpusBatchSize::PerRow
        } else {
            CorpusBatchSize::Fixed(batch_size.parse().unwrap_or(24))
        };
        if let Some(chunks) = data {
            if rowfile.is_some() || txtfile.is_some() {
                return Err(PyValueError::new_err(
                    "Pass either rowfile/txtfile or data=[...], not both.",
                ));
            }
            return Ok(Self {
                inner: corpus_from_memory(chunks, sort_rows_by_complexity, bs),
            });
        }
        // `use_two_files=False` deliberately builds an empty dataset (used to
        // inspect batch-size parsing); the default file mode requires sources.
        if use_two_files && rowfile.is_none() && txtfile.is_none() {
            return Err(PyValueError::new_err(
                "DatasetCorpus needs training data.\nFix: Pass rowfile=/txtfile= paths or an in-memory list like data=[\"some text chunk\"].",
            ));
        }
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

    /// Return corpus texts for training / hub finetune loops.
    fn as_texts(&self) -> Vec<String> {
        self.inner.rows.iter().map(|r| r.text.clone()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetCorpus(rows={}, format={:?}, type='corpus')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}
