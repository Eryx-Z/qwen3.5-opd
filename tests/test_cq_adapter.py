"""CPU tests use real native tiny Qwen dense + MoE conditional classes."""
import pytest
import torch
from transformers import (Qwen3_5Config, Qwen3_5ForConditionalGeneration,
                          Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration)

from cq_opd.model_adapter import QwenAdapter, attach_lora, check_logits_equivalence
from cq_opd.rollout import Rollout, build_batch, generate


def tiny(moe=False):
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, layer_types=["linear_attention", "full_attention"],
                linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=1, linear_num_value_heads=2,
                eos_token_id=2, pad_token_id=2, attention_dropout=0.0)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                  out_hidden_size=16, num_position_embeddings=16)
    if moe:
        text.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=8,
                    shared_expert_intermediate_size=8)
        config = Qwen3_5MoeConfig(text_config=text, vision_config=vision)
        model = Qwen3_5MoeForConditionalGeneration(config)
    else:
        config = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True)
        model = Qwen3_5ForConditionalGeneration(config)
    model.to(dtype=torch.bfloat16)
    attach_lora(model, ["model.language_model.layers.0.linear_attn.out_proj",
                        "model.language_model.layers.1.self_attn.q_proj"], rank=2, alpha=4)
    return QwenAdapter(model, trainable=True, gradient_checkpointing=True)


def records():
    return [Rollout([4, 5, 7, 2], 2, [7, 2], "", "a", False, 1),
            Rollout([4, 6, 8], 2, [8], "", "b", True, 2)]


@pytest.mark.parametrize("moe", [False, True])
def test_real_hidden_head_gradient(moe):
    torch.manual_seed(10)
    torch.set_num_threads(1)
    adapter = tiny(moe)
    batch = build_batch(records(), "cpu", 2)
    metrics = check_logits_equivalence(adapter, batch, atol=0.005, rtol=0.005)
    assert metrics["logits_max_abs"] < 0.005
    h = adapter.last_hidden(batch, with_grad=True)
    loss = adapter.project(h[:, :-1]).float().square().mean()
    params = adapter.lora_named_parameters()
    gradients = torch.autograd.grad(loss, [p for _, p in params], allow_unused=False)
    assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for _, p in params)
    assert not adapter.model.lm_head.weight.requires_grad
    with pytest.raises(ValueError, match="state"):
        adapter.last_hidden(batch, False, use_cache=True)


def test_right_padding_keeps_actual_eos():
    b = build_batch(records(), "cpu", 2)
    assert b["input_ids"][1, -1] == 2
    assert b["response_mask"].tolist() == [[False, False, True, True], [False, False, True, False]]
    valid = b["response_mask"][:, 1:] & b["attention_mask"][:, 1:]
    assert valid.tolist() == [[False, True, True], [False, True, False]]


class Tokenizer:
    eos_token_id = pad_token_id = 2
    bos_token_id = 1

    def decode(self, ids, **kwargs):
        return str(ids)


def test_generation_same_instance_rng_cache_and_no_fake_eos():
    torch.set_num_threads(1)
    a = tiny()
    a.model.config.text_config.eos_token_id = 100  # impossible: budget truncates
    state = torch.random.get_rng_state().clone()
    first = generate(a, Tokenizer(), [4, 5], 3, 111, id="x")
    second = generate(a, Tokenizer(), [4, 5], 3, 111, id="x")
    assert torch.equal(state, torch.random.get_rng_state())
    assert first == second
    assert first.truncated and len(first.response_ids) == 3
    assert first.input_ids == [4, 5] + first.response_ids
    assert a.language_model.training
    assert a.model.model.rope_deltas is None


def test_fresh_generation_config_and_greedy(monkeypatch):
    torch.set_num_threads(1)
    a = tiny()
    a.model.generation_config.top_k = 1
    a.model.generation_config.repetition_penalty = 2.0
    original = a.model.generate
    seen = []

    def spy(**kwargs):
        seen.append(kwargs["generation_config"])
        return original(**kwargs)

    monkeypatch.setattr(a.model, "generate", spy)
    generate(a, Tokenizer(), [4, 5], 2, 101)
    assert seen[-1].top_k == 0 and seen[-1].top_p == 1.0
    assert seen[-1].repetition_penalty == 1.0
    one = generate(a, Tokenizer(), [4, 5], 2, 102, do_sample=False)
    two = generate(a, Tokenizer(), [4, 5], 2, 103, do_sample=False)
    assert one.response_ids == two.response_ids
    assert not seen[-1].do_sample


def test_generated_eos_is_retained():
    torch.set_num_threads(1)
    a = tiny()
    a.model.config.text_config.eos_token_id = list(range(32))
    r = generate(a, Tokenizer(), [4, 5], 3, 201)
    assert len(r.response_ids) == 1 and not r.truncated
    assert r.input_ids[-1] == r.response_ids[-1]
    assert build_batch([r], "cpu", 2)["response_mask"].sum() == 1


def test_manifest_rejects_vision():
    a = tiny()
    with pytest.raises(ValueError, match="language linear"):
        attach_lora(a.model, ["model.visual.patch_embed.proj"])
