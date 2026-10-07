"""On-policy sampling from the same native student instance."""
from dataclasses import dataclass
from numbers import Integral

import torch
from transformers import GenerationConfig


@dataclass
class Rollout:
    input_ids: list[int]
    prompt_len: int
    response_ids: list[int]
    response_text: str
    id: str
    truncated: bool
    seed: int
    generation_batch_size: int = 1
    sample_index: int = 0
    policy_sha256: str | None = None
    sampled_log_probs: list[float] | None = None


def generate(adapter, tokenizer, prompt_ids, max_new_tokens, seed, *, id="", do_sample=True):
    if not prompt_ids or max_new_tokens < 1:
        raise ValueError("Need nonempty prompt and positive generation budget")
    model = adapter.model
    eos = model.config.text_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    # Fresh config, not model.generation_config: no inherited suppress/forced
    # tokens, top-p, repetition bias, temperature, or extra processors.
    sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 0} if do_sample else {}
    config = GenerationConfig(do_sample=do_sample, **sampling,
                              repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                              eos_token_id=eos, pad_token_id=pad, use_cache=True,
                              bos_token_id=tokenizer.bos_token_id)
    ids = torch.tensor([prompt_ids], device=adapter.device, dtype=torch.long)
    devices = [adapter.device.index if adapter.device.index is not None else torch.cuda.current_device()] \
        if adapter.device.type == "cuda" else []
    flags = [(module, module.training) for module in model.modules()]
    # A private seeded stream per call, restoring caller's CPU/CUDA RNG states.
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if devices:
                with torch.cuda.device(adapter.device):
                    torch.cuda.manual_seed(seed)
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    generation_config=config, past_key_values=None,
                                    return_dict_in_generate=False)
        sampled = output[0, len(prompt_ids):].tolist()
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        # Sequential B=1 generation preserves genuine EOS; no padding removal
        # based on ID (pad and EOS can coincide), and no fake EOS at truncation.
        ended = bool(sampled and sampled[-1] in eos_ids)
        return Rollout(list(prompt_ids) + sampled, len(prompt_ids), sampled,
                       tokenizer.decode(sampled, skip_special_tokens=True), str(id),
                       len(sampled) == max_new_tokens and not ended, int(seed))
    finally:
        for module, flag in flags:
            module.training = flag
        # HF generation cache is local and not returned; multimodal rope state
        # must not survive a new independent text-only question.
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None


def generate_batch(adapter, tokenizer, prompt_ids_list, max_new_tokens, seed, *,
                   ids=None, do_sample=True):
    """One native generation call using a shared batch seed, not per-row seeds.

    Provenance is ``native_batch_seed_plus_row``: seed, generation_batch_size,
    and sample_index bind each result to its native multinomial batch trajectory.
    Qwen computes text/multimodal position IDs from the left-padding mask itself.
    """
    if (not isinstance(max_new_tokens, Integral) or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1):
        raise ValueError("Need positive integer generation budget")
    if not isinstance(prompt_ids_list, (list, tuple)) or not prompt_ids_list:
        raise ValueError("Need nonempty prompt list")
    prompts = []
    for prompt in prompt_ids_list:
        if (not isinstance(prompt, (list, tuple)) or not prompt
                or any(not isinstance(token, Integral) or isinstance(token, bool)
                       or token < 0 for token in prompt)):
            raise ValueError("Need nonempty prompts of nonnegative integer token IDs")
        prompts.append([int(token) for token in prompt])
    batch_size = len(prompts)
    if ids is None:
        ids = [""] * batch_size
    elif (not isinstance(ids, (list, tuple)) or len(ids) != batch_size
          or any(not isinstance(value, str) for value in ids)):
        raise ValueError("ids must contain one string per prompt")
    model = adapter.model
    eos = model.config.text_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Need tokenizer pad_token_id or eos_token_id for left padding")
    sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 0} if do_sample else {}
    config = GenerationConfig(do_sample=do_sample, **sampling,
                              repetition_penalty=1.0, max_new_tokens=int(max_new_tokens),
                              num_beams=1, num_return_sequences=1,
                              eos_token_id=eos, pad_token_id=pad, use_cache=True,
                              bos_token_id=tokenizer.bos_token_id)
    width = max(map(len, prompts))
    input_ids = torch.full((batch_size, width), pad, device=adapter.device, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        input_ids[row, -len(prompt):] = torch.tensor(prompt, device=adapter.device, dtype=torch.long)
        attention[row, -len(prompt):] = 1
    devices = [adapter.device.index if adapter.device.index is not None else torch.cuda.current_device()] \
        if adapter.device.type == "cuda" else []
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            # Seed only CPU and this adapter's CUDA stream, not every GPU.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(adapter.device):
                    torch.cuda.manual_seed(seed)
            output = model.generate(input_ids=input_ids, attention_mask=attention,
                                    generation_config=config, past_key_values=None,
                                    return_dict_in_generate=False)
            if not isinstance(output, torch.Tensor):
                raise TypeError("Native generate must return a sequences tensor")
            if (output.ndim != 2 or output.shape[0] != batch_size
                    or not width < output.shape[1] <= width + max_new_tokens
                    or output.dtype != torch.long
                    or not torch.equal(output[:, :width], input_ids)):
                raise ValueError("Malformed native generation sequences")
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            results = []
            for row, prompt in enumerate(prompts):
                sampled = output[row, width:].tolist()
                ended = False
                for index, token in enumerate(sampled):
                    if token in eos_ids:
                        sampled = sampled[:index + 1]
                        ended = True
                        break
                # Only post-EOS positions are padding. A PAD ID before EOS is a
                # genuine sample (including when PAD == EOS); never invent EOS.
                results.append(Rollout(prompt + sampled, len(prompt), sampled,
                                       tokenizer.decode(sampled, skip_special_tokens=True), ids[row],
                                       len(sampled) == max_new_tokens and not ended, int(seed),
                                       generation_batch_size=batch_size, sample_index=row))
            return results
    finally:
        for module, flag in flags:
            module.training = flag
        if hasattr(model.model, "rope_deltas"):
            model.model.rope_deltas = None


def build_batch(rollouts, device, pad_id):
    if not rollouts or pad_id is None:
        raise ValueError("Need nonempty rollout list and explicit pad_id")
    length = max(len(r.input_ids) for r in rollouts)
    ids = torch.full((len(rollouts), length), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    response = torch.zeros_like(ids, dtype=torch.bool)
    for row, record in enumerate(rollouts):
        if not 0 < record.prompt_len <= len(record.input_ids):
            raise ValueError("Invalid prompt length")
        if record.input_ids[record.prompt_len:] != record.response_ids:
            raise ValueError("Response does not match actual sampled input suffix")
        n = len(record.input_ids)
        ids[row, :n] = torch.tensor(record.input_ids, dtype=torch.long, device=device)
        attention[row, :n] = True
        response[row, record.prompt_len:n] = True
    return {"input_ids": ids, "attention_mask": attention, "response_mask": response}
