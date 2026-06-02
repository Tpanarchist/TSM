from PIL import Image
import torch

from tsm.config import DatasetConfig, TsmConfig
from tsm.data import (
    ContestedTemporalObjectPermanenceDataset,
    ImageStreamDataset,
    MultiModeSyntheticImageStreamDataset,
    TemporalObjectPermanenceDataset,
    _contested_position,
    make_dataset,
)


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
    assert item["mode"].item() == 7
    assert torch.isfinite(item["image_tp1"]).all()


def test_multimode_synthetic_dataset_exposes_known_modes():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = MultiModeSyntheticImageStreamDataset(cfg, length=8, seed=11)

    modes = [ds[i]["mode"].item() for i in range(8)]

    assert modes == [0, 1, 2, 3, 0, 1, 2, 3]
    assert ds[0]["image_t"].shape == (1, 28, 28)
    assert ds[0]["image_tp1"].shape == (1, 28, 28)


def test_multimode_synthetic_test_split_uses_heldout_variants():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    train = MultiModeSyntheticImageStreamDataset(cfg, length=8, seed=11, split="train")
    heldout = MultiModeSyntheticImageStreamDataset(cfg, length=8, seed=11, split="test")

    assert torch.equal(train[1]["image_t"], heldout[1]["image_t"])
    assert not torch.equal(train[1]["image_tp1"], heldout[1]["image_tp1"])


def test_temporal_object_dataset_exposes_occlusion_phases():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = TemporalObjectPermanenceDataset(cfg, length=10, seed=13)

    modes = [ds[i]["mode"].item() for i in range(5)]
    visible = [ds[i]["visible_t"].item() for i in range(5)]
    occluded = [ds[i]["occluded_t"].item() for i in range(5)]

    assert modes == [0, 1, 2, 2, 3]
    assert visible == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert occluded == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert ds[2]["unexpected_disappearance"].item() == 1.0
    assert ds[4]["visible_tp1"].item() == 1.0
    assert ds[4]["identity_preserved"].item() == 1.0
    assert ds[0]["image_t"].shape == (1, 28, 28)
    assert ds[0]["object_position_t"].shape == (2,)
    assert ds.sequential


def test_contested_temporal_object_dataset_exposes_two_target_tracks():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = ContestedTemporalObjectPermanenceDataset(cfg, length=12, seed=13)

    phases = [ds[i]["phase"].item() for i in range(6)]
    tracks = [ds[i]["track_id"].item() for i in range(6)]
    file_ids = [ds[i]["object_file_id"].item() for i in range(2)]

    assert phases == [0, 0, 1, 1, 2, 2]
    assert tracks == [0, 1, 0, 1, 0, 1]
    assert file_ids == [0, 1]
    assert ds[0]["sequence_id"].item() == ds[1]["sequence_id"].item()
    assert ds[0]["object_id"].item() == ds[1]["object_id"].item()
    assert ds[6]["occluded_t"].item() == 1.0
    assert ds[8]["visible_tp1"].item() == 1.0
    assert ds[8]["same_class_contested"].item() == 1.0
    assert ds[0]["image_t"].shape == (1, 28, 28)
    assert ds.sequential


def test_contested_reappeared_trajectories_are_separable():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    for split in ("train", "test", "heldout"):
        distances = []
        for scene_id in range(16):
            phase = 0
            pos_a = torch.tensor(_contested_position(cfg, scene_id, 0, phase, split), dtype=torch.float32)
            pos_b = torch.tensor(_contested_position(cfg, scene_id, 1, phase, split), dtype=torch.float32)
            distances.append(torch.linalg.vector_norm(pos_a - pos_b))
        assert torch.stack(distances).min().item() > 8.0


def test_make_dataset_supports_temporal_object_alias():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = make_dataset(DatasetConfig(name="object_permanence", split="train", limit=7, seed=3), cfg)

    assert isinstance(ds, TemporalObjectPermanenceDataset)
    assert len(ds) == 7


def test_make_dataset_supports_contested_temporal_object_alias():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = make_dataset(
        DatasetConfig(name="temporal_objects_contested_position", split="train", limit=7, seed=3),
        cfg,
    )

    assert isinstance(ds, ContestedTemporalObjectPermanenceDataset)
    assert len(ds) == 7
