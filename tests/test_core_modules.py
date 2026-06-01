import torch

from tsm.config import TsmConfig
from tsm.context import ContextRouter
from tsm.definitions import DefinitionBank
from tsm.gate import MutationGate
from tsm.memory import Memory
from tsm.sae import SAE
from tsm.types import CandidateTruth, DriveState


def test_sae_shapes():
    cfg = TsmConfig(d_model=16, workspace_latents=8, contexts=3, definitions_per_context=4, attention_heads=4)
    sae = SAE(cfg)
    obs = torch.randn(2, 8, 16)
    expected = torch.randn(2, 8, 16)
    source_conf = torch.ones(2, 8, 1)
    attach_power = torch.zeros(2, 8, 16)
    drives = DriveState.zeros(2, "cpu")
    ctx = torch.softmax(torch.randn(2, 3), dim=-1)

    out = sae(obs, expected, source_conf, attach_power, drives, ctx)

    assert out.eps.shape == (2, 8, 16)
    assert out.severity.shape == (2, 8)
    assert out.coherence.shape == (2, 8)
    assert out.group.shape == (2, 8)


def test_context_router_probabilities_sum_to_one():
    cfg = TsmConfig(d_model=16, workspace_latents=8, contexts=5, definitions_per_context=4, attention_heads=4)
    router = ContextRouter(cfg)
    out = router(torch.randn(3, 8, 16), torch.randn(3, 8, 16))

    assert out.logits.shape == (3, 5)
    assert out.embedding.shape == (3, 16)
    assert torch.allclose(out.probs.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_gate_rejects_and_accepts_candidates():
    gate = MutationGate(min_evidence=0.75, min_coherence=0.6, max_risk=0.5)

    rejected = gate.consider(CandidateTruth("weak", evidence=0.2, coherence=0.9))
    accepted = gate.consider(CandidateTruth("stable", evidence=0.9, coherence=0.8))

    assert not rejected.accepted
    assert rejected.reason == "insufficient evidence"
    assert accepted.accepted
    assert accepted.reason == "accepted"


def test_memory_reads_prior_visible_object_trace_in_batch_order():
    memory = Memory()
    feature = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
    ])
    batch = {
        "sequence_id": torch.tensor([7, 7, 7], dtype=torch.long),
        "visible_t": torch.tensor([1.0, 0.0, 0.0]),
    }

    read = memory.read_write_object_files(batch, feature, step=3)

    assert read.write.tolist() == [True, False, False]
    assert read.hit.tolist() == [False, True, True]
    assert torch.allclose(read.feature[1], feature[0])
    assert torch.allclose(read.feature[2], feature[0])
    assert read.confidence[1].item() == 1.0


def test_definition_bank_can_project_memory_trace():
    cfg = TsmConfig(
        d_model=4,
        workspace_latents=2,
        contexts=1,
        definitions_per_context=2,
        attention_heads=1,
        memory_definition_scale=1.0,
    )
    bank = DefinitionBank(cfg)
    with torch.no_grad():
        bank.axes.zero_()
        bank.memory_axes.zero_()
        bank.memory_axes[0, 0, 0] = 1.0
        bank.log_tau.fill_(-4.0)
        bank.log_alpha.zero_()
    eps = torch.zeros(1, 2, 4)
    ctx = torch.ones(1, 1)
    memory_feature = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    memory_confidence = torch.ones(1, 1)

    base = bank.project(eps, ctx)
    conditioned = bank.project(eps, ctx, memory_feature, memory_confidence)

    assert base[0, 0].item() == 0.0
    assert conditioned[0, 0].item() > 0.0
