"""MagicMindNet grid architectures for DistribAI's MyTrainer sync.

DistribAI reads ``configs/grid_architectures.json`` from this tree
(``services_python/mytrainer_sync.py``); this module is the optional
Python-side builder for the same profiles. Requires ``pip install
magicmindnet`` — DistribAI itself never imports this file.
"""

from magicmindnet.distribai import chatbot_from_architecture, grid_architectures


def build_model(name, *, vocab_size=256, seed=None, **overrides):
    """Instantiate the named grid profile as a magicmindnet.Chatbot."""
    configs = grid_architectures()
    if name not in configs:
        raise ValueError(f"unknown grid architecture {name!r}; known: {sorted(configs)}")
    config = dict(configs[name])
    config.update(overrides)
    return chatbot_from_architecture(config, vocab_size=vocab_size, seed=seed)
