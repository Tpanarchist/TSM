import pytest

from tsm.config import DatasetConfig, TrainConfig, TsmConfig
from tsm.trainer import train


@pytest.mark.online
def test_online_mnist_twenty_steps(tmp_path):
    cfg = TrainConfig(
        run_name="online_mnist_test",
        model=TsmConfig(d_model=32, workspace_latents=8, contexts=3, definitions_per_context=4, image_size=28),
        dataset=DatasetConfig(name="mnist", split="train", cache_dir=str(tmp_path / "hf"), limit=64),
        batch_size=8,
        max_steps=20,
        log_interval=5,
        checkpoint_interval=20,
        sample_interval=20,
        runs_dir=str(tmp_path / "runs"),
    )
    run_dir = train(cfg, device_name="cpu")

    assert (run_dir / "checkpoints" / "latest.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
