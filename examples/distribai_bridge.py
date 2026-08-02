"""DistribAI compatibility bridge — run MagicMindNet on a DistribAI grid.

Everything here works offline (no orchestrator needed): detect a local
DistribAI install, map a Chatbot to DistribAI's ``architecture_config``,
build a worker script package, and exchange checkpoints in DistribAI's
native ``state_dict`` layout.

Run from the repo root after building (`maturin develop --release`):

    python examples/distribai_bridge.py

With a DistribAI orchestrator running on this device
(``python -m services_python.orchestrator_grpc``), submit for real:

    from magicmindnet.distribai import DistribAIClient
    client = DistribAIClient()          # ORCHESTRATOR_ADMIN_URL / 127.0.0.1:8766
    client.submit_training_job(bot, dataset=[{"input": "hi", "output": "yo"}])
"""

import tempfile
from pathlib import Path

import magicmindnet as ai
from magicmindnet import distribai as bridge


def main() -> None:
    # 1. Is DistribAI on this device? (repo checkout, pip package, node data)
    report = bridge.check_compatibility()
    install = report["install"]
    print(f"Python {report['python_version']} OK for both projects: {report['python_ok']}")
    print(f"DistribAI install detected: {install if install else 'none (bridge still works)'}")
    print(f"Orchestrator admin URL a client would use: {report['admin_url']}")

    # 2. Map a Chatbot onto DistribAI's declarative architecture_config.
    #    final_norm=True mirrors DistribAI's always-present final LayerNorm.
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=64, n_loops=2, seed=42, final_norm=True)
    config = bridge.architecture_config(bot)
    print(f"\narchitecture_config: {config}")
    print(f"(n_loops={bot.n_loops} became n_logical_layers={config['n_logical_layers']})")

    # ... and back: the config rebuilds the same shape anywhere.
    rebuilt = bridge.chatbot_from_architecture(config, vocab_size=256, seed=42)
    print(f"Rebuilt from config: n_layer={rebuilt.n_layer} n_loops={rebuilt.n_loops}")

    # 3. The MyTrainer sync contract: grid profiles DistribAI can register.
    profiles = bridge.grid_architectures()
    print(f"\nGrid profiles for external/mytrainer sync: {sorted(profiles)}")

    with tempfile.TemporaryDirectory() as tmp:
        tree = bridge.export_mytrainer_tree(Path(tmp) / "mytrainer")
        print("MyTrainer tree written:", [Path(p).name for p in tree])

        # 4. Build the script package a DistribAI worker node executes.
        package = bridge.build_script_package(
            config={"job_type": "train", "architecture_config": config},
            dataset=[{"input": "hi", "output": "hello there!"}],
        )
        meta = bridge.validate_script_package(package)
        print(f"\nScript package: {meta['size_bytes']} bytes, "
              f"{meta['member_count']} files, sha256={bridge.package_sha256(package)[:16]}...")

        # 5. Checkpoints in DistribAI's native state_dict naming — both ways.
        data = ai.DatasetQA(data=[{"input": "hi", "output": "hello there!"}])
        bot.train(data, epochs=2)
        checkpoint = Path(tmp) / "distribai_checkpoint.pt"
        named = bridge.export_checkpoint(bot, str(checkpoint))
        print(f"\nExported {len(named)} DistribAI tensors, e.g. "
              f"{sorted(named)[:2]} ... (torch.load-compatible)")

        # Loop count lives in the job's architecture_config (not the weights),
        # so hand it back on import — exactly like DistribAI's own workers do.
        back = bridge.import_checkpoint(str(checkpoint), n_loops=bot.n_loops)
        original, restored = bot.compute_mean_loss(data), back.compute_mean_loss(data)
        print(f"Loss parity after roundtrip: {original:.6f} -> {restored:.6f}")
        assert abs(original - restored) < 1e-4


if __name__ == "__main__":
    main()
