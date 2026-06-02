from PIL import Image
import torch

from tsm.config import DatasetConfig, TsmConfig
from tsm.data import (
    ContestedTemporalObjectPermanenceDataset,
    ImageStreamDataset,
    MultiModeSyntheticImageStreamDataset,
    TemporalObjectPermanenceDataset,
    _contested_curved_position,
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


def _wrapped_pair_assignment_accuracy(cfg: TsmConfig, position_fn, split: str) -> float:
    margin = max(4, cfg.image_size // 7)
    span = max(1, cfg.image_size - 2 * margin)
    correct = 0
    total = 0
    for scene_id in range(16):
        for target_track in (0, 1):
            file_predictions = []
            true_positions = []
            for track_id in (target_track, 1 - target_track):
                p1 = torch.tensor(position_fn(cfg, scene_id, track_id, 1, split), dtype=torch.float32)
                p2 = torch.tensor(position_fn(cfg, scene_id, track_id, 2, split), dtype=torch.float32)
                velocity = p2 - p1
                projected = ((p2 + velocity * 3.0 - margin) % span) + margin
                file_predictions.append(projected)
                true_positions.append(torch.tensor(position_fn(cfg, scene_id, track_id, 0, split), dtype=torch.float32))
            query = torch.stack(true_positions)
            files = torch.stack(file_predictions)
            diff = (query.unsqueeze(1) - files.unsqueeze(0)).abs()
            diff = torch.minimum(diff, (span - diff).abs())
            assignment = diff.square().sum(dim=-1).sqrt().argmin(dim=1)
            correct += int(assignment[0].item() == 0 and assignment[1].item() == 1)
            total += 1
    return correct / max(1, total)


def test_contested_curved_trajectories_are_separable_but_not_ballistic_oracle():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    for split in ("train", "test", "heldout"):
        distances = []
        for scene_id in range(16):
            pos_a = torch.tensor(_contested_curved_position(cfg, scene_id, 0, 0, split), dtype=torch.float32)
            pos_b = torch.tensor(_contested_curved_position(cfg, scene_id, 1, 0, split), dtype=torch.float32)
            distances.append(torch.linalg.vector_norm(pos_a - pos_b))
        assert torch.stack(distances).min().item() > 8.0
        assert _wrapped_pair_assignment_accuracy(cfg, _contested_curved_position, split) < 0.75


def test_contested_curved_dataset_exposes_nonlinear_motion_flag():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = ContestedTemporalObjectPermanenceDataset(cfg, length=12, seed=13, motion="curved")

    item = ds[8]

    assert item["same_class_contested"].item() == 1.0
    assert item["nonlinear_contested_motion"].item() == 1.0
    assert item["visible_tp1"].item() == 1.0
    assert item["distractor_position_tp1"].shape == (2,)
    assert item["all_object_positions_tp1"].shape == (2, 2)
    assert item["all_object_file_ids"].shape == (2,)


def test_contested_curved_dataset_supports_three_and_four_tracks():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    for track_count in (3, 4):
        ds = ContestedTemporalObjectPermanenceDataset(
            cfg,
            length=track_count * 5,
            seed=13,
            motion="curved",
            track_count=track_count,
        )
        item = ds[track_count * 4]

        assert ds.track_count == track_count
        assert item["contested_track_count"].item() == track_count
        assert item["all_object_positions_tp1"].shape == (track_count, 2)
        assert item["all_object_file_ids"].shape == (track_count,)
        assert torch.unique(item["all_object_file_ids"]).numel() == track_count


def test_contested_curved_wide_four_track_spacing_control():
    narrow = TsmConfig(d_model=16, image_size=28, image_channels=1)
    wide = TsmConfig(d_model=16, image_size=40, image_channels=1)
    narrow_distances = []
    wide_distances = []
    for scene_id in range(16):
        narrow_positions = torch.tensor(
            [_contested_curved_position(narrow, scene_id, track_id, 0, "test", 4) for track_id in range(4)],
            dtype=torch.float32,
        )
        wide_positions = torch.tensor(
            [_contested_curved_position(wide, scene_id, track_id, 0, "test", 4) for track_id in range(4)],
            dtype=torch.float32,
        )
        narrow_pairwise = torch.cdist(narrow_positions.unsqueeze(0), narrow_positions.unsqueeze(0)).squeeze(0)
        wide_pairwise = torch.cdist(wide_positions.unsqueeze(0), wide_positions.unsqueeze(0)).squeeze(0)
        eye = torch.eye(4, dtype=torch.bool)
        narrow_distances.append(narrow_pairwise.masked_select(~eye).min())
        wide_distances.append(wide_pairwise.masked_select(~eye).min())

    assert torch.stack(wide_distances).mean().item() > torch.stack(narrow_distances).mean().item() + 4.0
    assert torch.stack(wide_distances).min().item() > 12.0


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


def test_make_dataset_supports_contested_curved_temporal_object_alias():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    ds = make_dataset(
        DatasetConfig(name="temporal_objects_contested_curved", split="train", limit=7, seed=3),
        cfg,
    )

    assert isinstance(ds, ContestedTemporalObjectPermanenceDataset)
    assert ds.motion == "curved"
    assert len(ds) == 7


def test_make_dataset_supports_contested_curved_count_aliases():
    cfg = TsmConfig(d_model=16, image_size=28, image_channels=1)
    for name, track_count in (
        ("temporal_objects_contested_curved_3", 3),
        ("temporal_objects_contested_curved_4", 4),
        ("temporal_objects_contested_curved_4_wide", 4),
    ):
        ds = make_dataset(DatasetConfig(name=name, split="train", limit=7, seed=3), cfg)

        assert isinstance(ds, ContestedTemporalObjectPermanenceDataset)
        assert ds.motion == "curved"
        assert ds.track_count == track_count
        assert ds.dataset_name == name
