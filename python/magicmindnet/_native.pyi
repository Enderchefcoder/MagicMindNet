"""Type stubs for the Rust-native `magicmindnet._native` module."""

from typing import Any

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CPUError(Exception): ...
class CUDAError(Exception): ...
class DataMismatchError(Exception): ...
class DataMissingRowError(Exception): ...
class ModelMismatchError(Exception): ...

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DatasetQA:
    def __init__(
        self,
        file: str | None = None,
        user_row: str = "input",
        ai_row: str = "output",
        system_row: str | None = None,
        image_row: str = "image",
        vision_patch_grid: int = 1,
        multipleturn: bool = True,
        tokenizer: str = "ChatXML",
        cot: bool = True,
        thinktag: str = "",
        data: list[dict[str, str]] | None = None,
    ) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def type_(self) -> str: ...
    @property
    def vision_patch_grid(self) -> int: ...
    def format_sample(self, index: int) -> str: ...
    def as_pairs(self) -> list[tuple[str, str]]: ...
    def sample_image_path(self, index: int) -> str | None: ...
    def sample_image_paths(self, index: int) -> list[str]: ...

class DatasetCorpus:
    def __init__(
        self,
        use_two_files: bool = True,
        rowfile: str | None = None,
        txtfile: str | None = None,
        sort_rows_by_complexity: bool = True,
        rows_with_corpus_chunk: str = "text",
        batch_size: str = "row",
        data: list[str] | None = None,
    ) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def type_(self) -> str: ...
    @property
    def corpus_batch_size(self) -> str: ...
    def as_texts(self) -> list[str]: ...

class DatasetClassification:
    def __init__(
        self,
        file: str | None = None,
        text_col: str = "text",
        tags_col: str = "label",
        data: list[dict[str, str]] | None = None,
    ) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def type_(self) -> str: ...
    def unique_labels(self) -> list[str]: ...
    def as_pairs(self) -> list[tuple[str, str]]: ...

class DatasetImageGen:
    def __init__(self, file: str) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def type_(self) -> str: ...
    def resolve_image_path(self, rel: str) -> str: ...
    def image_path_at(self, index: int) -> str: ...
    def prompt_at(self, index: int) -> str: ...

class DatasetImageEdit:
    def __init__(self, file: str) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def format(self) -> str: ...
    @property
    def type_(self) -> str: ...
    def resolve_image_path(self, rel: str) -> str: ...
    def resolve_mask_path(self, rel: str) -> str: ...
    def image_path_at(self, index: int) -> str: ...
    def mask_path_at(self, index: int) -> str: ...
    def prompt_at(self, index: int) -> str: ...

# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class BytePairEncoder:
    @staticmethod
    def train(texts: list[str], vocab_size: int = 512, num_merges: int = 32) -> BytePairEncoder: ...
    @staticmethod
    def train_from_qa(
        dataset: DatasetQA, vocab_size: int = 512, num_merges: int = 32
    ) -> BytePairEncoder: ...
    @staticmethod
    def train_from_corpus(
        dataset: DatasetCorpus, vocab_size: int = 512, num_merges: int = 32
    ) -> BytePairEncoder: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> BytePairEncoder: ...
    @property
    def merge_count(self) -> int: ...
    @property
    def vocab_size(self) -> int: ...

class UnigramEncoder:
    @staticmethod
    def train(texts: list[str], vocab_size: int = 512) -> UnigramEncoder: ...
    @staticmethod
    def train_from_qa(dataset: DatasetQA, vocab_size: int = 512) -> UnigramEncoder: ...
    @staticmethod
    def train_from_corpus(dataset: DatasetCorpus, vocab_size: int = 512) -> UnigramEncoder: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> UnigramEncoder: ...
    def prune_pieces_below_logprob(self, min_log_prob: float) -> None: ...
    @property
    def piece_count(self) -> int: ...
    @property
    def vocab_size(self) -> int: ...

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

class TrainConfig:
    epochs: int
    batch_size: int
    cuda: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    lr_schedule: str
    warmup_steps: int
    verbose: bool
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 8,
        cuda: bool = False,
        optimizer: str = "hybrid",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        lr_schedule: str = "constant",
        warmup_steps: int = 0,
        verbose: bool = False,
    ) -> None: ...

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Chatbot:
    def __init__(
        self,
        vision: bool = False,
        autoset: str | None = None,
        vocab_size: int = 32000,
        n_layer: int | None = None,
        d_model: int | None = None,
        seed: int | None = None,
        use_learned_pos_embed: bool = False,
        max_seq_len: int = 512,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        n_heads: int | None = None,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        ffn_dim: int | None = None,
        n_loops: int = 1,
        tie_embeddings: bool = False,
        norm: str = "layer",
        ffn: str = "gelu",
        loop_embed: bool = False,
        final_norm: bool = False,
        lora_rank: int = 0,
        coda_layers: int = 0,
        prelude_layers: int = 0,
        max_loops: int | None = None,
        attention_window: int | None = None,
    ) -> None: ...
    @property
    def parameters(self) -> int: ...
    @property
    def layer_size(self) -> int: ...
    @property
    def tokenizer(self) -> str: ...
    @property
    def has_vision(self) -> bool: ...
    @property
    def has_vision_patch_encoder(self) -> bool: ...
    @property
    def has_vision_rgb_conv(self) -> bool: ...
    @property
    def has_vision_cross_attn(self) -> bool: ...
    @property
    def vision_patch_dim(self) -> int: ...
    @property
    def vision_rgb_dim(self) -> int: ...
    @property
    def uses_causal_attention(self) -> bool: ...
    @property
    def use_learned_pos_embed(self) -> bool: ...
    @property
    def use_rope(self) -> bool: ...
    @property
    def rope_theta(self) -> float: ...
    @property
    def max_seq_len(self) -> int: ...
    @property
    def init_seed(self) -> int | None: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def n_layer(self) -> int: ...
    @property
    def d_model(self) -> int: ...
    @property
    def n_heads(self) -> int: ...
    @property
    def n_kv_heads(self) -> int: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def ffn_dim(self) -> int: ...
    @property
    def n_loops(self) -> int: ...
    @property
    def tie_embeddings(self) -> bool: ...
    @property
    def norm(self) -> str: ...
    @property
    def ffn(self) -> str: ...
    @property
    def loop_embed(self) -> bool: ...
    @property
    def final_norm(self) -> bool: ...
    @property
    def lora_rank(self) -> int: ...
    @property
    def coda_layers(self) -> int: ...
    @property
    def prelude_layers(self) -> int: ...
    @property
    def max_loops(self) -> int: ...
    @property
    def attention_window(self) -> int | None: ...
    def save(
        self,
        path: str,
        format: str = "safetensors",
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
    ) -> None: ...
    @staticmethod
    def load(path: str, format: str | None = None) -> Chatbot: ...
    def train(
        self,
        dataset: DatasetQA | DatasetCorpus,
        config: TrainConfig | None = None,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        cuda: bool | None = None,
        verbose: bool | None = None,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
    ) -> list[float]: ...
    def chat(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.1,
        stop_strings: list[str] | None = None,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
    ) -> str: ...
    def compute_loss(
        self,
        input: str,
        target: str,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
        image_patch: list[float] | None = None,
        image_patches: list[list[float]] | None = None,
    ) -> float: ...
    def compute_mean_loss(
        self,
        dataset: DatasetQA | DatasetCorpus,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
    ) -> float: ...
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        typical_p: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_kv_cache: bool = True,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
        image_patch: list[float] | None = None,
        image_patches: list[list[float]] | None = None,
        stop_token_ids: list[int] | None = None,
        stop_strings: list[str] | None = None,
        json_mode: bool = False,
        grammar: str | None = None,
    ) -> str: ...
    def generate_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        typical_p: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_kv_cache: bool = True,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
        image_patch: list[float] | None = None,
        image_patches: list[list[float]] | None = None,
        stop_token_ids: list[int] | None = None,
        stop_strings: list[str] | None = None,
        json_mode: bool = False,
        grammar: str | None = None,
    ) -> list[str]: ...
    def generate_tokens(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        typical_p: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_kv_cache: bool = True,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
        image_patch: list[float] | None = None,
        image_patches: list[list[float]] | None = None,
        stop_token_ids: list[int] | None = None,
        stop_strings: list[str] | None = None,
        json_mode: bool = False,
        grammar: str | None = None,
    ) -> list[int]: ...
    def embed(
        self,
        texts: str | list[str],
        *,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
    ) -> list[float] | list[list[float]]: ...
    def chat_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        typical_p: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_kv_cache: bool = True,
        bpe_encoder: BytePairEncoder | None = None,
        unigram_encoder: UnigramEncoder | None = None,
        gpt2_encoder: Gpt2BpeEncoder | None = None,
        stop_token_ids: list[int] | None = None,
        stop_strings: list[str] | None = None,
        json_mode: bool = False,
        grammar: str | None = None,
    ) -> str: ...

def format_chat_messages(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str: ...

class Classifier:
    def __init__(self, num_labels: int, input_dim: int, seed: int | None = None) -> None: ...
    @staticmethod
    def with_labels(labels: list[str], input_dim: int, seed: int | None = None) -> Classifier: ...
    @staticmethod
    def from_classification(
        ds: DatasetClassification, input_dim: int, seed: int | None = None
    ) -> Classifier: ...
    @property
    def labels(self) -> list[str]: ...
    @property
    def init_seed(self) -> int | None: ...
    @property
    def input_dim(self) -> int: ...
    @property
    def num_labels(self) -> int: ...
    def predict(self, text: str) -> dict[str, float]: ...
    def predict_label(self, text: str) -> str: ...
    def compute_loss(self, text: str, label: str) -> float: ...
    def compute_mean_loss(self, dataset: DatasetClassification) -> float: ...
    def save(self, path: str, format: str = "safetensors") -> None: ...
    @staticmethod
    def load(path: str, format: str | None = None) -> Classifier: ...
    def train(
        self,
        dataset: DatasetClassification,
        config: TrainConfig | None = None,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        cuda: bool | None = None,
        verbose: bool | None = None,
    ) -> list[float]: ...

class Diffusion:
    def __init__(self) -> None: ...
    @property
    def latent_channels(self) -> int: ...
    @property
    def parameters(self) -> int: ...
    def smoke_step(self) -> bool: ...
    def denoise_loss(self, t: int) -> float: ...
    def denoise_loss_on_image(self, path: str, t: int) -> float: ...
    def denoise_loss_on_image_masked(self, image_path: str, mask_path: str, t: int) -> float: ...
    def sample_rgb_patch(self, steps: int = 8, seed: int | None = None) -> list[float]: ...
    def sample_inpaint_rgb_patch(
        self, image_path: str, mask_path: str, steps: int = 8, seed: int | None = None
    ) -> list[float]: ...
    def sample_rgb_patch_to_png(self, path: str, steps: int = 8, seed: int | None = None) -> None: ...
    def sample_inpaint_rgb_patch_to_png(
        self,
        path: str,
        image_path: str,
        mask_path: str,
        steps: int = 8,
        seed: int | None = None,
    ) -> None: ...
    def compute_mean_denoise_loss(
        self, dataset: DatasetImageGen | DatasetImageEdit, t: int = 7
    ) -> float: ...
    def save(self, path: str, format: str = "safetensors") -> None: ...
    @staticmethod
    def load(path: str, format: str | None = None) -> Diffusion: ...
    def train(
        self,
        dataset: DatasetImageGen | DatasetImageEdit,
        config: TrainConfig | None = None,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        cuda: bool | None = None,
        verbose: bool | None = None,
    ) -> list[float]: ...

# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def Train(
    model: Chatbot,
    dataset: DatasetQA | DatasetCorpus,
    config: TrainConfig,
    bpe_encoder: BytePairEncoder | None = None,
    unigram_encoder: UnigramEncoder | None = None,
    gpt2_encoder: Gpt2BpeEncoder | None = None,
) -> list[float]: ...
def TrainClassifier(
    model: Classifier, dataset: DatasetClassification, config: TrainConfig
) -> list[float]: ...
def TrainDiffusion(
    model: Diffusion, dataset: DatasetImageGen | DatasetImageEdit, config: TrainConfig
) -> list[float]: ...
def RL(
    model: Chatbot,
    dataset: DatasetQA,
    train_config: TrainConfig,
    reward_amount: float,
    punishment_amount: float,
    rl_type: str = "policy",
    bpe_encoder: BytePairEncoder | None = None,
    unigram_encoder: UnigramEncoder | None = None,
    gpt2_encoder: Gpt2BpeEncoder | None = None,
) -> None: ...
def SPIN(
    model: Chatbot,
    selfplay_epochs: int,
    dataset: DatasetQA,
    bpe_encoder: BytePairEncoder | None = None,
    unigram_encoder: UnigramEncoder | None = None,
    gpt2_encoder: Gpt2BpeEncoder | None = None,
) -> None: ...

# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------

def export(
    model: Chatbot,
    format: str,
    path: str,
    bpe_encoder: BytePairEncoder | None = None,
    unigram_encoder: UnigramEncoder | None = None,
) -> None: ...
def import_model(format: str, files: list[str]) -> Chatbot: ...
def load_checkpoint(path: str) -> Any: ...
def load(path: str) -> Any: ...
def merge(model1: Chatbot, model2: Chatbot) -> Chatbot: ...
def quantize(model: Chatbot, quant: str) -> None: ...
def export_classifier_model(model: Classifier, format: str, path: str) -> None: ...
def import_classifier_model(format: str, files: list[str]) -> Classifier: ...
def merge_classifier(model1: Classifier, model2: Classifier) -> Classifier: ...
def quantize_classifier_model(model: Classifier, quant: str) -> None: ...
def export_diffusion_model(model: Diffusion, format: str, path: str) -> None: ...
def import_diffusion_model(format: str, files: list[str]) -> Diffusion: ...
def merge_diffusion_model(model1: Diffusion, model2: Diffusion) -> Diffusion: ...
def quantize_diffusion_model(model: Diffusion, quant: str) -> None: ...

# ---------------------------------------------------------------------------
# Global array IO (NumPy .npy/.npz, PyTorch .pt) — all from scratch
# ---------------------------------------------------------------------------

def read_npy(path: str) -> tuple[list[int], list[float]]: ...
def write_npy(path: str, shape: list[int], data: list[float], dtype: str = "f4") -> None: ...
def read_npz(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_npz(
    path: str,
    arrays: list[tuple[str, list[int], list[float]]],
    compress: bool = False,
    dtype: str = "f4",
) -> None: ...
def read_pt(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_pt(path: str, arrays: list[tuple[str, list[int], list[float]]]) -> None: ...
def read_h5(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def read_keras(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def read_tf_checkpoint(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def read_onnx(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_onnx(path: str, arrays: list[tuple[str, list[int], list[float]]]) -> None: ...
def read_safetensors(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_safetensors(
    path: str, arrays: list[tuple[str, list[int], list[float]]], dtype: str = "f32"
) -> None: ...
def read_flax(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_flax(path: str, arrays: list[tuple[str, list[int], list[float]]]) -> None: ...
def read_gguf_arrays(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_gguf_arrays(
    path: str, arrays: list[tuple[str, list[int], list[float]]], dtype: str = "f32"
) -> None: ...
def read_arrays_auto(path: str) -> tuple[str, list[tuple[str, list[int], list[float]]]]: ...
def read_arrays_auto_bytes(path: str) -> tuple[str, list[tuple[str, list[int], bytes]]]: ...
def write_arrays_bytes(
    path: str, format: str, entries: list[tuple[str, list[int], bytes]]
) -> None: ...
def read_tflite(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def read_pickle_arrays(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def read_zarr(path: str) -> list[tuple[str, list[int], list[float]]]: ...
def write_zarr(
    path: str, arrays: list[tuple[str, list[int], list[float]]], zarr_format: int = 2
) -> None: ...
def write_safetensors_sharded(
    index_path: str,
    arrays: list[tuple[str, list[int], list[float]]],
    max_shard_bytes: int,
) -> None: ...
def write_pickle_arrays(path: str, arrays: list[tuple[str, list[int], list[float]]]) -> None: ...
def read_ggml_legacy(
    path: str,
) -> tuple[str, list[int], list[tuple[bytes, float]], list[tuple[str, list[int], list[float]]]]: ...
def write_tf_checkpoint(path: str, arrays: list[tuple[str, list[int], list[float]]]) -> None: ...
def write_h5(
    path: str, arrays: list[tuple[str, list[int], list[float]]], compress: bool = False
) -> None: ...
def gguf_info_json(path: str) -> str: ...
def load_gguf_tokenizer(path: str) -> UnigramEncoder: ...
def load_gguf_bpe_tokenizer(path: str) -> Gpt2BpeEncoder: ...
def dequantize_ggml(type_name: str, data: list[int], numel: int) -> list[float]: ...

class Gpt2BpeEncoder:
    @staticmethod
    def from_vocab(tokens: list[str], merges: list[str] = []) -> Gpt2BpeEncoder: ...
    @staticmethod
    def from_hf_tokenizer_json(path: str) -> Gpt2BpeEncoder: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def token(self, id: int) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Resources & vision helpers
# ---------------------------------------------------------------------------

def limit_resources(percent: str) -> None: ...
def limit(percent: str) -> None: ...
def limit_percent() -> int: ...
def vision_rgb_patch_from_image_path(path: str) -> list[float]: ...
def vision_rgb_patches_from_image_path(path: str, grid: int = 1) -> list[list[float]]: ...
def vision_rgb_patch_from_image_path_py(path: str) -> list[float]: ...
def vision_rgb_patches_from_image_path_py(path: str, grid: int = 1) -> list[list[float]]: ...
