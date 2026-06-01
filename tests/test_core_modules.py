import torch

from tsm.config import TsmConfig
from tsm.context import ContextRouter
from tsm.gate import MutationGate
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
