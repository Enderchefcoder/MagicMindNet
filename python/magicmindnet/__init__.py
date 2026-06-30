"""MagicMindNet — easy, powerful AI with a from-scratch Rust core.

Usage::

    import magicmindnet as ai

    data = ai.DatasetQA(file="qa.json", user_row="input", ai_row="output")
    bot = ai.Chatbot(autoset="sub-100M")
    cfg = ai.TrainConfig(epochs=1, batch_size=4, cuda=False, optimizer="hybrid")
    ai.Train(bot, data, cfg)
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
    merge,
    merge_classifier,
    merge_diffusion_model,
    quantize,
    quantize_classifier_model,
    quantize_diffusion_model,
)
from magicmindnet.bpe_io import load_bpe_sidecar
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
    "import_classifier",
    "import_classifier_model",
    "import_diffusion",
    "import_diffusion_model",
    "import_model",
    "limit",
    "limit_percent",
    "load_bpe_sidecar",
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
