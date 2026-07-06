"""MagicMindNet — easy, powerful AI with a from-scratch Rust core.

Train your first chatbot in five lines::

    import magicmindnet as ai

    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello!"}])
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    bot.train(data, epochs=3)
    print(bot.chat("hi"))

Save and reload any model with one call each::

    bot.save("bot.mmn")
    bot = ai.load("bot.mmn")

See docs/getting_started.md for the full beginner tutorial.
"""

from magicmindnet._native import (
    RL,
    SPIN,
    BytePairEncoder,
    Chatbot,
    Classifier,
    CPUError,
    CUDAError,
    DataMismatchError,
    DataMissingRowError,
    DatasetClassification,
    DatasetCorpus,
    DatasetImageEdit,
    DatasetImageGen,
    DatasetQA,
    Diffusion,
    Gpt2BpeEncoder,
    ModelMismatchError,
    Train,
    TrainClassifier,
    TrainConfig,
    TrainDiffusion,
    UnigramEncoder,
    export,
    export_classifier_model,
    export_diffusion_model,
    import_classifier_model,
    import_diffusion_model,
    import_model,
    limit,
    limit_percent,
    load,
    load_checkpoint,
    merge,
    merge_classifier,
    merge_diffusion_model,
    quantize,
    quantize_classifier_model,
    quantize_diffusion_model,
)
from magicmindnet.bpe_io import load_bpe_sidecar
from magicmindnet.interop import (
    gguf_info,
    load_flax,
    load_gguf_bpe_tokenizer,
    load_gguf_tokenizer,
    load_h5,
    load_keras,
    load_npy,
    load_npz,
    load_onnx,
    load_pt,
    load_safetensors,
    load_tf_checkpoint,
    save_flax,
    save_h5,
    save_npy,
    save_npz,
    save_onnx,
    save_pt,
    save_safetensors,
    save_tf_checkpoint,
)
from magicmindnet.unigram_io import load_unigram_sidecar
from magicmindnet.vision import (
    VISION_PATCH_DIM,
    VISION_RGB_CHANNELS,
    VISION_RGB_DIM,
    VISION_RGB_SPATIAL,
    vision_patch_from_text,
    vision_rgb_patch_from_image_path,
    vision_rgb_patch_from_text,
    vision_rgb_patches_from_image_path,
)

# Public aliases matching chatbot IO naming
export_classifier = export_classifier_model
import_classifier = import_classifier_model
quantize_classifier = quantize_classifier_model
export_diffusion = export_diffusion_model
import_diffusion = import_diffusion_model
merge_diffusion = merge_diffusion_model
quantize_diffusion = quantize_diffusion_model

__all__ = [
    "__version__",
    "CPUError",
    "CUDAError",
    "BytePairEncoder",
    "UnigramEncoder",
    "Chatbot",
    "Classifier",
    "DataMismatchError",
    "DataMissingRowError",
    "DatasetClassification",
    "DatasetCorpus",
    "DatasetImageEdit",
    "DatasetImageGen",
    "DatasetQA",
    "Diffusion",
    "ModelMismatchError",
    "TrainConfig",
    "Train",
    "TrainClassifier",
    "TrainDiffusion",
    "RL",
    "SPIN",
    "export",
    "export_classifier",
    "export_classifier_model",
    "export_diffusion",
    "export_diffusion_model",
    "gguf_info",
    "Gpt2BpeEncoder",
    "import_classifier",
    "import_classifier_model",
    "import_diffusion",
    "import_diffusion_model",
    "import_model",
    "limit",
    "limit_percent",
    "load",
    "load_bpe_sidecar",
    "load_checkpoint",
    "load_flax",
    "load_gguf_bpe_tokenizer",
    "load_gguf_tokenizer",
    "load_h5",
    "load_keras",
    "load_npy",
    "load_npz",
    "load_onnx",
    "load_pt",
    "load_safetensors",
    "load_tf_checkpoint",
    "load_unigram_sidecar",
    "merge",
    "merge_classifier",
    "merge_diffusion",
    "merge_diffusion_model",
    "quantize",
    "quantize_classifier",
    "quantize_classifier_model",
    "quantize_diffusion",
    "quantize_diffusion_model",
    "save_flax",
    "save_h5",
    "save_npy",
    "save_npz",
    "save_onnx",
    "save_pt",
    "save_safetensors",
    "save_tf_checkpoint",
    "VISION_PATCH_DIM",
    "VISION_RGB_CHANNELS",
    "VISION_RGB_DIM",
    "VISION_RGB_SPATIAL",
    "vision_patch_from_text",
    "vision_rgb_patch_from_image_path",
    "vision_rgb_patches_from_image_path",
    "vision_rgb_patch_from_text",
]

__version__ = "0.1.0"
