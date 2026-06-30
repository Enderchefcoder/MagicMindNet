use crate::chatxml::ChatXmlConfig;
use crate::error::{data_missing_row, data_mismatch};
use mmn_core::MmnError;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DatasetType {
    Qa,
    Corpus,
    Classification,
    ImageGen,
    ImageEdit,
}

pub struct DatasetMeta {
    pub rows: usize,
    pub format: String,
    pub dataset_type: DatasetType,
}

pub struct DatasetQA {
    pub meta: DatasetMeta,
    pub samples: Vec<QaSample>,
    pub chatxml: ChatXmlConfig,
    /// Parent directory of `cfg.file` for resolving relative image paths.
    pub source_dir: std::path::PathBuf,
    /// Tile grid for single-image multi-patch prefix (from config).
    pub vision_patch_grid: usize,
}

#[derive(Clone)]
pub struct QaSample {
    pub input: String,
    pub output: String,
    pub system: Option<String>,
    /// Optional relative or absolute paths to input image files.
    pub image_paths: Vec<String>,
}

pub struct DatasetQAConfig {
    pub file: String,
    pub user_row: String,
    pub ai_row: String,
    pub system_row: Option<String>,
    /// Optional JSON column for input image path(s) (default `"image"` when set).
    pub image_row: Option<String>,
    /// When a single image is loaded, split it into `grid×grid` 8×8 tiles (default 1).
    pub vision_patch_grid: usize,
    pub multiple_turn: bool,
    pub thinktag: String,
    pub cot: bool,
}

impl DatasetQA {
    pub fn load(cfg: DatasetQAConfig) -> Result<Self, MmnError> {
        let path = Path::new(&cfg.file);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json");
        let samples = match ext {
            "parquet" => load_qa_parquet(path, &cfg)?,
            "json" | "jsonl" => load_qa_json(path, &cfg)?,
            _ => load_qa_json(path, &cfg)?,
        };
        let meta = DatasetMeta {
            rows: samples.len(),
            format: ext.to_string(),
            dataset_type: DatasetType::Qa,
        };
        let source_dir = path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| Path::new(".").to_path_buf());
        Ok(Self {
            meta,
            samples,
            chatxml: ChatXmlConfig::from_thinktag(&cfg.thinktag, cfg.cot),
            source_dir,
            vision_patch_grid: cfg.vision_patch_grid.clamp(1, crate::vision::MAX_VISION_PATCH_GRID),
        })
    }

    pub fn validate_for_model(model_type: &str) -> Result<(), MmnError> {
        if model_type == "diffusion" {
            return Err(data_mismatch("qa", "diffusion"));
        }
        Ok(())
    }
}

fn load_qa_json(path: &Path, cfg: &DatasetQAConfig) -> Result<Vec<QaSample>, MmnError> {
    let text = fs::read_to_string(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    let rows: Vec<Value> = if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l))
            .collect::<Result<_, _>>()
            .map_err(|e| MmnError::Other {
                message: e.to_string(),
            })?
    } else {
        serde_json::from_str(&text).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?
    };
    load_qa_from_values(&rows, cfg)
}

fn load_qa_from_values(rows: &[Value], cfg: &DatasetQAConfig) -> Result<Vec<QaSample>, MmnError> {
    let mut samples = Vec::new();
    for row in rows {
        let obj = row.as_object().ok_or_else(|| MmnError::Other {
            message: "expected object rows".into(),
        })?;
        if !obj.contains_key(&cfg.user_row) {
            return Err(data_missing_row(&cfg.user_row));
        }
        if !obj.contains_key(&cfg.ai_row) {
            return Err(data_missing_row(&cfg.ai_row));
        }
        let input = obj[&cfg.user_row].as_str().unwrap_or("").to_string();
        let output = obj[&cfg.ai_row].as_str().unwrap_or("").to_string();
        let system = cfg
            .system_row
            .as_ref()
            .and_then(|k| obj.get(k).and_then(|v| v.as_str()).map(String::from));
        let image_paths = cfg.image_row.as_ref().and_then(|k| obj.get(k)).map(|v| {
            if let Some(s) = v.as_str() {
                crate::vision::parse_image_path_list(s)
            } else if let Some(arr) = v.as_array() {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            } else {
                Vec::new()
            }
        }).unwrap_or_default();
        samples.push(QaSample {
            input,
            output,
            system,
            image_paths,
        });
    }
    Ok(samples)
}

fn load_qa_parquet(path: &Path, cfg: &DatasetQAConfig) -> Result<Vec<QaSample>, MmnError> {
    use arrow::array::{Array, StringArray};
    let file = fs::File::open(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?
        .build()
        .map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
    let mut values: Vec<Value> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let schema = batch.schema();
        let n = batch.num_rows();
        for r in 0..n {
            let mut obj = serde_json::Map::new();
            for i in 0..batch.num_columns() {
                let name = schema.field(i).name().clone();
                let col = batch.column(i);
                let s = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(r).to_string())
                    .unwrap_or_else(|| format!("{:?}", col));
                obj.insert(name, Value::String(s));
            }
            values.push(Value::Object(obj));
        }
    }
    load_qa_from_values(&values, cfg)
}

pub struct CorpusRow {
    pub text: String,
    pub complexity: f32,
}

pub struct DatasetCorpus {
    pub meta: DatasetMeta,
    pub rows: Vec<CorpusRow>,
    pub batch_size: CorpusBatchSize,
}

pub enum CorpusBatchSize {
    PerRow,
    Fixed(usize),
}

pub struct DatasetCorpusConfig {
    pub use_two_files: bool,
    pub rowfile: Option<String>,
    pub txtfile: Option<String>,
    pub sort_rows_by_complexity: bool,
    pub rows_with_corpus_chunk: String,
    pub batch_size: CorpusBatchSize,
}

impl DatasetCorpus {
    pub fn load(cfg: DatasetCorpusConfig) -> Result<Self, MmnError> {
        let mut rows = Vec::new();
        if cfg.use_two_files {
            if let Some(ref rf) = cfg.rowfile {
                let text = fs::read_to_string(rf).map_err(|e| MmnError::Other {
                    message: e.to_string(),
                })?;
                let json: Vec<Value> = serde_json::from_str(&text).map_err(|e| MmnError::Other {
                    message: e.to_string(),
                })?;
                for v in json {
                    if let Some(t) = v.get(&cfg.rows_with_corpus_chunk).and_then(|x| x.as_str()) {
                        rows.push(CorpusRow {
                            text: t.to_string(),
                            complexity: t.len() as f32,
                        });
                    }
                }
            }
            if let Some(ref tf) = cfg.txtfile {
                let text = fs::read_to_string(tf).map_err(|e| MmnError::Other {
                    message: e.to_string(),
                })?;
                for chunk in text.split_whitespace().collect::<Vec<_>>().chunks(24) {
                    rows.push(CorpusRow {
                        text: chunk.join(" "),
                        complexity: chunk.len() as f32,
                    });
                }
            }
        }
        if cfg.sort_rows_by_complexity {
            rows.sort_by(|a, b| {
                a.complexity
                    .partial_cmp(&b.complexity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let meta = DatasetMeta {
            rows: rows.len(),
            format: "corpus".into(),
            dataset_type: DatasetType::Corpus,
        };
        Ok(Self {
            meta,
            rows,
            batch_size: cfg.batch_size,
        })
    }
}

pub struct DatasetClassification {
    pub meta: DatasetMeta,
    pub samples: Vec<(String, String)>,
}

impl DatasetClassification {
    pub fn load(path: &str, text_col: &str, tags_col: &str) -> Result<Self, MmnError> {
        let text = fs::read_to_string(path).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let rows: Vec<Value> = serde_json::from_str(&text).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let mut samples = Vec::new();
        let mut all_tags = std::collections::HashSet::new();
        for row in &rows {
            if let Some(obj) = row.as_object() {
                if let (Some(t), Some(tag)) = (
                    obj.get(text_col).and_then(|v| v.as_str()),
                    obj.get(tags_col).and_then(|v| v.as_str()),
                ) {
                    samples.push((t.to_string(), tag.to_string()));
                    all_tags.insert(tag.to_string());
                }
            }
        }
        if samples.is_empty() && !rows.is_empty() {
            for row in rows {
                if let Some(obj) = row.as_object() {
                    if let Some(t) = obj.get(text_col).and_then(|v| v.as_str()) {
                        let tag = format!("class_{}", samples.len());
                        samples.push((t.to_string(), tag));
                    }
                }
            }
        }
        let _ = all_tags;
        Ok(Self {
            meta: DatasetMeta {
                rows: samples.len(),
                format: "classification".into(),
                dataset_type: DatasetType::Classification,
            },
            samples,
        })
    }

    pub fn unique_labels(&self) -> Vec<String> {
        let mut tags: Vec<String> = self.samples.iter().map(|(_, t)| t.clone()).collect();
        tags.sort();
        tags.dedup();
        tags
    }
}

#[cfg(test)]
mod classification_tests {
    use super::*;

    #[test]
    fn unique_labels_sorted_deduped() {
        let ds = DatasetClassification {
            meta: DatasetMeta {
                rows: 2,
                format: "test".into(),
                dataset_type: DatasetType::Classification,
            },
            samples: vec![
                ("a".into(), "Happy".into()),
                ("b".into(), "Sad".into()),
                ("c".into(), "Happy".into()),
            ],
        };
        assert_eq!(ds.unique_labels(), vec!["Happy", "Sad"]);
    }
}

#[cfg(test)]
mod qa_tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, body: &str) -> String {
        let dir = std::env::temp_dir();
        let path = dir.join(name);
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn missing_ai_row_raises() {
        let path = write_temp("mmn_qa_no_output.json", r#"[{"input":"only"}]"#);
        let cfg = DatasetQAConfig {
            file: path,
            user_row: "input".into(),
            ai_row: "output".into(),
            system_row: None,
            image_row: None,
            vision_patch_grid: 1,
            multiple_turn: false,
            thinktag: "|".into(),
            cot: true,
        };
        let result = DatasetQA::load(cfg);
        let err = result.err().expect("expected missing output column");
        assert!(
            err.message().contains("output"),
            "expected missing output column, got: {}",
            err.message()
        );
    }

    #[test]
    fn jsonl_loads_two_rows() {
        let path = write_temp(
            "mmn_qa.jsonl",
            "{\"input\":\"a\",\"output\":\"b\"}\n{\"input\":\"c\",\"output\":\"d\"}\n",
        );
        let cfg = DatasetQAConfig {
            file: path,
            user_row: "input".into(),
            ai_row: "output".into(),
            system_row: None,
            image_row: None,
            vision_patch_grid: 1,
            multiple_turn: false,
            thinktag: "|".into(),
            cot: true,
        };
        let ds = DatasetQA::load(cfg).unwrap();
        assert_eq!(ds.meta.rows, 2);
        assert_eq!(ds.meta.format, "jsonl");
    }

    #[test]
    fn qa_loads_optional_image_path() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_qa_img.json");
        fs::write(
            &path,
            r#"[{"input":"describe","output":"red square","image":"pics/a.png"}]"#,
        )
        .unwrap();
        let cfg = DatasetQAConfig {
            file: path.to_string_lossy().into_owned(),
            user_row: "input".into(),
            ai_row: "output".into(),
            system_row: None,
            image_row: Some("image".into()),
            vision_patch_grid: 1,
            multiple_turn: false,
            thinktag: "|".into(),
            cot: true,
        };
        let ds = DatasetQA::load(cfg).unwrap();
        assert_eq!(ds.samples[0].image_paths, vec!["pics/a.png".to_string()]);
        assert_eq!(ds.source_dir, dir);
    }
}

#[cfg(test)]
mod classification_load_tests {
    use super::*;

    #[test]
    fn auto_tags_when_tag_column_missing() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_cls_auto.json");
        fs::write(&path, r#"[{"text":"one"},{"text":"two"}]"#).unwrap();
        let ds = DatasetClassification::load(path.to_str().unwrap(), "text", "tag").unwrap();
        assert_eq!(ds.samples.len(), 2);
        assert_eq!(ds.unique_labels(), vec!["class_0", "class_1"]);
    }
}

#[cfg(test)]
mod corpus_tests {
    use super::*;

    #[test]
    fn sort_by_complexity_orders_short_first() {
        let dir = std::env::temp_dir();
        let rowfile = dir.join("mmn_corp_sort.json");
        fs::write(
            &rowfile,
            r#"[{"text":"short"},{"text":"much longer row text"}]"#,
        )
        .unwrap();
        let cfg = DatasetCorpusConfig {
            use_two_files: true,
            rowfile: Some(rowfile.to_string_lossy().into_owned()),
            txtfile: None,
            sort_rows_by_complexity: true,
            rows_with_corpus_chunk: "text".into(),
            batch_size: CorpusBatchSize::PerRow,
        };
        let ds = DatasetCorpus::load(cfg).unwrap();
        assert!(ds.rows.len() >= 2);
        assert!(ds.rows[0].text.len() <= ds.rows[1].text.len());
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn image_gen_resolve_image_path_relative_to_manifest() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_img_resolve.json");
        fs::write(
            &path,
            r#"[{"prompt":"cat","image":"samples/cat.png"}]"#,
        )
        .unwrap();
        let ds = DatasetImageGen::load(path.to_str().unwrap()).unwrap();
        let resolved = ds.resolve_image_path("samples/cat.png");
        assert!(resolved.ends_with("samples/cat.png"));
    }

    #[test]
    fn image_gen_loads_negative_prompt() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_img_gen.json");
        fs::write(
            &path,
            r#"[{"prompt":"cat","image":"c.png","negative_prompt":"blur"}]"#,
        )
        .unwrap();
        let ds = DatasetImageGen::load(path.to_str().unwrap()).unwrap();
        assert_eq!(ds.samples.len(), 1);
        assert_eq!(ds.samples[0].negative_prompt.as_deref(), Some("blur"));
    }

    #[test]
    fn image_edit_loads_mask_and_negative_prompt() {
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_img_edit.json");
        fs::write(
            &path,
            r#"[{"prompt":"fix sky","image":"a.png","mask_image":"m.png","negative_prompt":"haze"}]"#,
        )
        .unwrap();
        let ds = DatasetImageEdit::load(path.to_str().unwrap()).unwrap();
        assert_eq!(ds.samples.len(), 1);
        assert_eq!(ds.samples[0].mask_image, "m.png");
        assert_eq!(ds.samples[0].negative_prompt.as_deref(), Some("haze"));
    }
}

pub struct ImageSample {
    pub prompt: String,
    pub image_path: String,
    pub negative_prompt: Option<String>,
}

pub struct DatasetImageGen {
    pub meta: DatasetMeta,
    pub samples: Vec<ImageSample>,
    pub manifest_path: String,
}

impl DatasetImageGen {
    pub fn load(manifest: &str) -> Result<Self, MmnError> {
        let text = fs::read_to_string(manifest).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let rows: Vec<Value> = serde_json::from_str(&text).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let mut samples = Vec::new();
        for row in rows {
            if let Some(obj) = row.as_object() {
                samples.push(ImageSample {
                    prompt: obj
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    image_path: obj
                        .get("image")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    negative_prompt: obj
                        .get("negative_prompt")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                });
            }
        }
        Ok(Self {
            meta: DatasetMeta {
                rows: samples.len(),
                format: "image_gen".into(),
                dataset_type: DatasetType::ImageGen,
            },
            samples,
            manifest_path: manifest.to_string(),
        })
    }

    pub fn resolve_image_path(&self, rel: &str) -> std::path::PathBuf {
        let base = std::path::Path::new(&self.manifest_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        base.join(rel)
    }
}

pub struct DatasetImageEdit {
    pub meta: DatasetMeta,
    pub samples: Vec<ImageEditSample>,
}

pub struct ImageEditSample {
    pub prompt: String,
    pub mask_image: String,
    pub image: String,
    pub negative_prompt: Option<String>,
}

impl DatasetImageEdit {
    pub fn load(manifest: &str) -> Result<Self, MmnError> {
        let text = fs::read_to_string(manifest).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let rows: Vec<Value> = serde_json::from_str(&text).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let mut samples = Vec::new();
        for row in rows {
            if let Some(obj) = row.as_object() {
                samples.push(ImageEditSample {
                    prompt: obj
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    mask_image: obj
                        .get("mask_image")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    image: obj
                        .get("image")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    negative_prompt: obj
                        .get("negative_prompt")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                });
            }
        }
        Ok(Self {
            meta: DatasetMeta {
                rows: samples.len(),
                format: "image_edit".into(),
                dataset_type: DatasetType::ImageEdit,
            },
            samples,
        })
    }
}
