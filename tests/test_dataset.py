from PIL import Image
import torch

from tsm.config import DatasetConfig, TsmConfig
from tsm.data import ImageStreamDataset, MultiModeSyntheticImageStreamDataset


def test_image_stream_dataset_uses_mock_rows_without_network():
    rows = [{"image": Image.new("L", (28, 28), color=128), "label": 7}]
    ds = ImageStreamDataset(
        DatasetConfig(name="mnist", split="train", cache_dir="data/hf", limit=None, seed=5),
        TsmConfig(d_model=16, image_size=28, image_channels=1),
        hf_dataset=rows,
    )

    item = ds[0]

    assert item["image_t"].shape == (1, 28, 28)
    assert item["image_tp1"].shape == (1, 28, 28)
    assert item["label"].item() == 7
    assert torch.isfinite(item["image_tp1"]).all()


def test_multimode_synthetic_dataset_exposes_known_modes():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = MultiModeSyntheticImageStreamDataset(cfg, length=8, seed=11)

    modes = [ds[i]["mode"].item() for i in range(8)]

    assert modes == [0, 1, 2, 3, 0, 1, 2, 3]
    assert ds[0]["image_t"].shape == (1, 28, 28)
    assert ds[0]["image_tp1"].shape == (1, 28, 28)
