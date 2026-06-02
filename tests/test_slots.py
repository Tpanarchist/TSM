import torch

from tsm.config import TsmConfig
from tsm.slots import ObjectSlotReadout


def test_object_slot_readout_separates_two_salient_regions():
    cfg = TsmConfig(
        image_size=16,
        object_slot_count=2,
        object_slot_sigma=1.5,
        object_slot_nms_radius=5.0,
        object_slot_salience_threshold=0.01,
    )
    readout = ObjectSlotReadout(cfg)
    image = torch.full((1, 1, 16, 16), 0.02)
    image[0, 0, 3:6, 3:6] = 0.9
    image[0, 0, 11:14, 11:14] = 0.8

    slots = readout(image)

    assert slots.state.shape == (1, 2, 8)
    assert slots.position.shape == (1, 2, 2)
    assert slots.local_images.shape == (1, 2, 1, 16, 16)
    assert slots.valid.all()
    distance = torch.cdist(slots.position[0], slots.position[0])
    assert distance[0, 1] > 6.0
    expected = torch.tensor([[4.0, 4.0], [12.0, 12.0]])
    assert torch.cdist(slots.position[0], expected).min(dim=1).values.mean() < 1.5


def test_object_slot_readout_can_be_disabled():
    cfg = TsmConfig(image_size=16, object_slot_count=0)
    readout = ObjectSlotReadout(cfg)

    slots = readout(torch.rand(2, 1, 16, 16))

    assert slots.state.shape == (2, 0, 8)
    assert slots.local_images.shape == (2, 0, 1, 16, 16)
