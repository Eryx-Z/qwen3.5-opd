"""CPU-only native batched generation, masks, and shared-seed provenance."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from cq_opd.model_adapter import QwenAdapter, attach_lora
from cq_opd.rollout import Rollout, build_batch, generate, generate_batch


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return str(ids)


class SpyModel(torch.nn.Module):
    def __init__(self, suffix=None, eos=2):
        super().__init__()
        self.model = torch.nn.Linear(2, 2)
        self.model.rope_deltas = torch.ones(1)
        self.config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=eos))
        self.generation_config = SimpleNamespace(top_k=1, top_p=.1, temperature=.3,
                                                 repetition_penalty=9, suppress_tokens=[3])
        self.suffix = suffix
        self.calls = []
        self.failure = None
        self.output_transform = lambda output: output
        self.model.eval()  # Deliberately heterogeneous module flags.

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in self.modules())
        torch.rand(3)  # Ensure RNG restoration is meaningful, even on exceptions.
        if self.failure:
            raise self.failure
        inputs = kwargs["input_ids"]
        if self.suffix is None:
            config = kwargs["generation_config"]
            probabilities = torch.full((inputs.shape[0], 32), 1 / 32)
            suffix = torch.stack([torch.multinomial(probabilities, 1).squeeze(1)
                                  for _ in range(config.max_new_tokens)], dim=1)
        else:
            suffix = torch.tensor(self.suffix, dtype=torch.long)
        return self.output_transform(torch.cat([inputs, suffix], dim=1))


def spy_adapter(suffix=None, eos=2):
    return SimpleNamespace(model=SpyModel(suffix, eos), device=torch.device("cpu"))


def snapshot(model):
    return (torch.random.get_rng_state().clone(),
            [(module, module.training) for module in model.modules()],
            [(p, p.detach().clone(), p.grad, None if p.grad is None else p.grad.clone())
             for p in model.parameters()])


def assert_restored(model, before):
    rng, flags, params = before
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(module.training == flag for module, flag in flags)
    for parameter, value, grad, grad_value in params:
        assert torch.equal(parameter, value)
        assert parameter.grad is grad
        if grad is not None:
            assert torch.equal(grad, grad_value)
    assert model.model.rope_deltas is None


def test_left_padding_single_native_call_config_and_provenance():
    adapter = spy_adapter([[7, 2, 0], [0, 9, 10], [2, 0, 0]])
    prompts = [[4, 5, 6], [0, 4], [5]]  # Actual PAD in a prompt stays attended.
    adapter.model.model.weight.grad = torch.ones_like(adapter.model.model.weight)
    before = snapshot(adapter.model)
    results = generate_batch(adapter, Tokenizer(), prompts, 3, 42, ids=["a", "b", "c"])
    assert_restored(adapter.model, before)
    assert len(adapter.model.calls) == 1
    call = adapter.model.calls[0]
    assert call["input_ids"].tolist() == [[4, 5, 6], [0, 0, 4], [0, 0, 5]]
    assert call["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
    assert "position_ids" not in call  # Native Qwen owns its 4D-axis positions.
    assert call["past_key_values"] is None and call["return_dict_in_generate"] is False
    config = call["generation_config"]
    assert (config.temperature, config.top_p, config.top_k, config.repetition_penalty) == (1., 1., 0, 1.)
    assert config.do_sample and config.use_cache and config.suppress_tokens is None
    assert config.forced_bos_token_id is None and config.forced_eos_token_id is None
    assert config.num_beams == 1 and config.num_return_sequences == 1
    assert [r.response_ids for r in results] == [[7, 2], [0, 9, 10], [2]]
    assert [r.truncated for r in results] == [False, True, False]
    for row, record in enumerate(results):
        assert record.input_ids == prompts[row] + record.response_ids
        assert record.prompt_len == len(prompts[row]) and record.id == "abc"[row]
        assert (record.seed, record.generation_batch_size, record.sample_index) == (42, 3, row)
        assert record.response_text == str(record.response_ids)
    batch = build_batch(results, "cpu", 0)
    assert batch["response_mask"].sum().item() == 6
    assert prompts == [[4, 5, 6], [0, 4], [5]]


@pytest.mark.parametrize("eos,pad,suffix,expected,truncated", [
    (2, 2, [[7, 2, 2], [8, 9, 10]], [[7, 2], [8, 9, 10]], [False, True]),
    ([2, 3], 0, [[0, 3, 2], [2, 0, 0]], [[0, 3], [2]], [False, False]),
    (2, 0, [[0, 0, 0], [7, 0, 8]], [[0, 0, 0], [7, 0, 8]], [True, True]),
    (None, 0, [[7, 2, 0], [2, 0, 0]], [[7, 2], [2]], [False, False]),
])
def test_eos_lists_padding_and_fallback(eos, pad, suffix, expected, truncated):
    adapter = spy_adapter(suffix, eos)
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = pad
    result = generate_batch(adapter, tokenizer, [[4], [5, 6]], 3, 12)
    assert [r.response_ids for r in result] == expected
    assert [r.truncated for r in result] == truncated
    assert [r.id for r in result] == ["", ""]


def test_missing_pad_uses_tokenizer_eos_not_model_eos():
    adapter = spy_adapter([[9, 3], [3, 2]], eos=3)
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = None
    results = generate_batch(adapter, tokenizer, [[4], [5, 6]], 2, 12)
    assert adapter.model.calls[0]["input_ids"].tolist() == [[2, 4], [5, 6]]
    assert [r.response_ids for r in results] == [[9, 3], [3]]


def test_no_eos_defined_does_not_strip_pad_or_invent_eos():
    adapter = spy_adapter([[0, 7], [8, 0]], eos=None)
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    result = generate_batch(adapter, tokenizer, [[4], [5]], 2, 12)
    assert [r.response_ids for r in result] == [[0, 7], [8, 0]]
    assert all(r.truncated for r in result)


def test_short_native_output_without_eos_is_not_marked_budget_truncated():
    result = generate_batch(spy_adapter([[7], [8]]), Tokenizer(), [[4], [5]], 3, 12)
    assert all(not r.truncated for r in result)
    assert [r.response_ids for r in result] == [[7], [8]]


def test_whole_batch_replay_and_multinomial_rows():
    adapter = spy_adapter(eos=99)
    before = snapshot(adapter.model)
    first = generate_batch(adapter, Tokenizer(), [[4]] * 16, 8, 42)
    assert_restored(adapter.model, before)
    second = generate_batch(adapter, Tokenizer(), [[4]] * 16, 8, 42)
    assert_restored(adapter.model, before)
    assert first == second
    assert len({tuple(r.response_ids) for r in first}) > 1
    # No assertion of equivalence to sixteen individually seeded trajectories.
    assert all(r.seed == 42 and r.generation_batch_size == 16 for r in first)


@pytest.mark.parametrize("failure", ["generate", "decode"])
def test_exceptions_restore_rng_module_flags_parameters_and_gradients(failure):
    adapter = spy_adapter([[7, 2], [9, 2]])
    adapter.model.model.weight.grad = torch.ones_like(adapter.model.model.weight)
    tokenizer = Tokenizer()
    if failure == "generate":
        adapter.model.failure = RuntimeError("generation failure")
    else:
        def failing_decode(*args, **kwargs):
            torch.rand(3)
            raise RuntimeError("decode failure")
        tokenizer.decode = failing_decode
    before = snapshot(adapter.model)
    with pytest.raises(RuntimeError, match="failure"):
        generate_batch(adapter, tokenizer, [[4], [5]], 2, 12)
    assert_restored(adapter.model, before)


@pytest.mark.parametrize("transform,error", [
    (lambda output: None, TypeError),
    (lambda output: SimpleNamespace(sequences=output), TypeError),
    (lambda output: output.tolist(), TypeError),
    (lambda output: output[0], ValueError),
    (lambda output: output[:1], ValueError),
    (lambda output: output[:, :1], ValueError),
    (lambda output: torch.cat([output, output[:, -1:]], dim=1), ValueError),
    (lambda output: output.float(), ValueError),
    (lambda output: output + 1, ValueError),
])
def test_native_return_validation_and_restoration(transform, error):
    adapter = spy_adapter([[7, 2], [9, 2]])
    adapter.model.output_transform = transform
    before = snapshot(adapter.model)
    with pytest.raises(error):
        generate_batch(adapter, Tokenizer(), [[4], [5]], 2, 12)
    assert_restored(adapter.model, before)


@pytest.mark.parametrize("prompts,budget,ids", [
    ([], 2, None), (None, 2, None), ([[]], 2, None), ([[4], []], 2, None),
    ([None], 2, None), (["4"], 2, None), ([[1.5]], 2, None), ([[True]], 2, None),
    ([[-1]], 2, None), ([[4]], 0, None), ([[4]], -1, None), ([[4]], 1.5, None),
    ([[4]], True, None), ([[4]], 2, []), ([[4]], 2, "x"), ([[4]], 2, [4]),
])
def test_invalid_requests_fail_before_native_call(prompts, budget, ids):
    adapter = spy_adapter()
    before = snapshot(adapter.model)
    with pytest.raises(ValueError):
        generate_batch(adapter, Tokenizer(), prompts, budget, 12, ids=ids)
    assert not adapter.model.calls
    assert torch.equal(before[0], torch.random.get_rng_state())
    assert all(module.training == flag for module, flag in before[1])


def test_missing_pad_and_eos_refused():
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id = None
    adapter = spy_adapter()
    with pytest.raises(ValueError, match="pad_token_id or eos_token_id"):
        generate_batch(adapter, tokenizer, [[4], [5, 6]], 2, 12)
    assert not adapter.model.calls


@pytest.fixture
def native():
    torch.set_num_threads(1)
    # Fixed fixture initialization, unrelated to rollout seed selection.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                    head_dim=8, layer_types=["linear_attention", "full_attention"],
                    linear_key_head_dim=8, linear_value_head_dim=8,
                    linear_num_key_heads=1, linear_num_value_heads=2,
                    eos_token_id=2, pad_token_id=0, attention_dropout=0.)
        vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                      out_hidden_size=16, num_position_embeddings=16)
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=text, vision_config=vision, tie_word_embeddings=True))
        attach_lora(model, ["model.language_model.layers.0.linear_attn.out_proj",
                            "model.language_model.layers.1.self_attn.q_proj"], rank=2, alpha=4)
        adapter = QwenAdapter(model, trainable=True, gradient_checkpointing=True)
    return adapter


def test_real_qwen_whole_batch_replay_no_student_parameter_or_grad_mutation(native, monkeypatch):
    native.model.config.text_config.eos_token_id = 99  # Impossible EOS: exact budget.
    params = native.lora_named_parameters()
    params[0][1].grad = torch.full_like(params[0][1], .37)
    before = snapshot(native.model)
    calls = []
    original = native.model.generate

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(native.model, "generate", spy)
    prompts = [[4], [4, 5], [6, 7, 8]]
    first = generate_batch(native, Tokenizer(), prompts, 4, 111)
    assert_restored(native.model, before)
    second = generate_batch(native, Tokenizer(), prompts, 4, 111)
    assert_restored(native.model, before)
    assert first == second and len(calls) == 2
    assert all(r.truncated and len(r.response_ids) == 4 for r in first)
    assert [r.sample_index for r in first] == [0, 1, 2]


def test_real_qwen_native_positions_greedy_matches_unpadded_single(native, monkeypatch):
    native.model.config.text_config.eos_token_id = 99
    native.model.generation_config.suppress_tokens = list(range(32))
    native.model.generation_config.repetition_penalty = 9
    forwards = []
    original = native.model.forward

    def spy(*args, **kwargs):
        seen = {key: value.detach().clone() for key, value in kwargs.items()
                if key in ("input_ids", "attention_mask", "position_ids")}
        output = original(*args, **kwargs)
        seen["logits"] = output.logits.detach().clone()
        forwards.append(seen)
        return output

    # Preserve the native signature used by HF generation's kwarg validation.
    import functools
    monkeypatch.setattr(native.model, "forward", functools.wraps(original)(spy))
    prompts = [[4], [5, 6], [7, 8, 9]]
    before = snapshot(native.model)
    batch = generate_batch(native, Tokenizer(), prompts, 3, 111, do_sample=False)
    assert_restored(native.model, before)
    first = forwards[0]
    assert first["input_ids"].shape == first["attention_mask"].shape == (3, 3)
    assert first["attention_mask"].tolist() == [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    assert first["position_ids"].shape == (4, 3, 3)
    expected = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 2]])
    for axis in first["position_ids"]:
        assert torch.equal(axis, expected)
    assert all(call["position_ids"].shape == (4, 3, 1) for call in forwards[1:])
    batch_logits = [call["logits"][:, -1] for call in forwards]
    singles = [generate(native, Tokenizer(), prompt, 3, 222, do_sample=False) for prompt in prompts]
    assert [r.response_ids for r in batch] == [r.response_ids for r in singles]
    # Beyond matching argmax, left padding preserves native per-step logits.
    for row in range(len(prompts)):
        for step in range(3):
            single_logits = forwards[3 + row * 3 + step]["logits"][0, -1]
            torch.testing.assert_close(batch_logits[step][row], single_logits,
                                       atol=1e-6, rtol=1e-5)
    replay = generate_batch(native, Tokenizer(), prompts, 3, 999, do_sample=False)
    assert [r.response_ids for r in replay] == [r.response_ids for r in batch]
    assert all(r.generation_batch_size == 1 and r.sample_index == 0 for r in singles)


def test_real_qwen_actual_eos_list_stops_without_fabrication(native):
    native.model.config.text_config.eos_token_id = list(range(32))
    before = snapshot(native.model)
    records = generate_batch(native, Tokenizer(), [[4], [5, 6]], 3, 201)
    assert_restored(native.model, before)
    assert all(len(r.response_ids) == 1 and not r.truncated for r in records)
    assert all(r.input_ids[-1] == r.response_ids[-1] for r in records)
    assert build_batch(records, "cpu", 0)["response_mask"].sum().item() == 2


def test_legacy_rollout_provenance_defaults():
    record = Rollout([4, 2], 1, [2], "", "a", False, 42)
    assert record.generation_batch_size == 1 and record.sample_index == 0
