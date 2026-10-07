"""Verified Transformers 5.12.1 Qwen3.5 conditional-model split adapter.

No PEFT wrapper or second student: native generate and hidden forwards share
exactly the same modules. LoRA master weights and their arithmetic are FP32.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

FP8_KERNEL_REVISION = "061130fedf845f320c56de4425f7404f6512c87e"  # upstream v2
FP8_KERNEL_SOURCE_HASH = "d99336493e5877af04016d91f6f974b43c47b6e3562bc58705af2e12e428ad5c"

import torch
from torch import nn
from torch.nn import functional as F


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def require_gpu_guard():
    if not os.environ.get("VERL_OPD_GUARD_RUN"):
        raise RuntimeError("GPU model loading requires diagnostics/guard.py (VERL_OPD_GUARD_RUN)")


class FP32LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                             dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                             dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x):
        base = self.base(x)
        # Explicitly disable ambient autocast for FP32 LoRA master arithmetic.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
        return base + delta.to(base.dtype)


def _conditional_class(config):
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration
    if config.model_type == "qwen3_5":
        return Qwen3_5ForConditionalGeneration
    if config.model_type == "qwen3_5_moe":
        return Qwen3_5MoeForConditionalGeneration
    raise TypeError(f"Unsupported Qwen conditional model_type: {config.model_type}")


class QwenAdapter:
    def __init__(self, model, *, trainable=False, gradient_checkpointing=False, source_path=None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
        if not isinstance(model, _conditional_class(model.config)):
            raise TypeError("Expected native Qwen3.5 conditional-generation model")
        self.model = model
        self.language_model = model.model.language_model
        if not isinstance(self.language_model, (Qwen3_5TextModel, Qwen3_5MoeTextModel)):
            raise TypeError("Unverified language_model class")
        if not hasattr(self.language_model, "norm") or not isinstance(model.lm_head, nn.Linear):
            raise TypeError("Unverified final norm/output projection")
        self.trainable = trainable
        self.gradient_checkpointing = gradient_checkpointing
        self.source_path = str(source_path or model.config._name_or_path)
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
        if gradient_checkpointing:
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.set_scoring_mode()

    @property
    def device(self):
        return self.language_model.embed_tokens.weight.device

    def set_scoring_mode(self):
        # Keep this flag in place until the graph has been backwarded: checkpoint
        # recomputation must see the same mode. Both FD sides use the same mode.
        self.model.eval()
        self.language_model.train(self.trainable and self.gradient_checkpointing)

    def last_hidden(self, batch, with_grad, use_cache=False):
        if use_cache:
            raise ValueError("Scoring/training must not retain recurrent/KV state")
        self.set_scoring_mode()
        with torch.set_grad_enabled(with_grad):
            return self.language_model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"],
                                       past_key_values=None, use_cache=False,
                                       output_hidden_states=False, return_dict=True).last_hidden_state

    def project(self, rows):
        # Installed vLLM defaults its LM-head OUTPUT to FP32 even with BF16
        # weights/backbone. Match that policy instead of rounding logits first.
        if getattr(self,'fp32_head_output',False) and rows.dtype==torch.bfloat16:
            weight=self.model.lm_head.weight
            # CUDA out_dtype mm has no autograd derivative in this torch build.
            # Differentiable scoring must use the equivalent FP32 projection.
            if rows.is_cuda and not (torch.is_grad_enabled() and rows.requires_grad):
                result=torch.mm(rows.reshape(-1,rows.shape[-1]),weight.t(),out_dtype=torch.float32)
                return result.reshape(*rows.shape[:-1],weight.shape[0])
            return F.linear(rows.float(),weight.float())
        return self.model.lm_head(rows)

    @contextmanager
    def scoped_gradient_projector(self):
        """Reuse one frozen FP32 head conversion within a chunked hidden VJP.

        Lazy allocation keeps empty selections cheap. No model tensor is changed;
        this scope must finish before any FD precision/weight mutation. Returned
        projectors expire at exit, including exceptions, releasing the snapshot.
        """
        weight = self.model.lm_head.weight
        signature = (weight.data_ptr(), weight._version, weight.dtype)
        converted = None
        active = True

        def project(rows):
            nonlocal converted
            if not active:
                raise RuntimeError('gradient projector used outside its scope')
            if (self.model.lm_head.weight is not weight or
                    (weight.data_ptr(), weight._version, weight.dtype) != signature):
                raise RuntimeError('head changed during gradient projector scope')
            if (getattr(self, 'fp32_head_output', False) and rows.dtype == torch.bfloat16
                    and torch.is_grad_enabled() and rows.requires_grad):
                if weight.requires_grad:
                    raise ValueError('scoped gradient projector requires a frozen head')
                if converted is None:
                    converted = weight.detach().float()
                return F.linear(rows.float(), converted)
            return self.project(rows)

        try:
            yield project
        finally:
            active = False
            converted = None

    def lora_named_parameters(self):
        return [(name, p) for name, p in self.model.named_parameters()
                if name.endswith((".lora_A", ".lora_B"))]

    def metadata(self):
        config = self.model.config.to_dict()
        return {"path": self.source_path, "revision": getattr(self.model.config, "_commit_hash", None),
                "config_hash": fingerprint(config),
                "quantization_hash": fingerprint(config.get("quantization_config")),
                "config_file_hash": _file_hash(Path(self.source_path) / "config.json"),
                "class": type(self.model).__name__, "language_class": type(self.language_model).__name__,
                "lora_names_hash": fingerprint([n for n, _ in self.lora_named_parameters()]),
                "lora_manifest_hash": getattr(self, "manifest_hash", None),
                "lora_config": getattr(self, "lora_config", None),
                "precision": {"embedding_dtype": str(self.language_model.embed_tokens.weight.dtype),
                              "head_dtype": str(self.model.lm_head.weight.dtype),
                              "head_output_fp32":getattr(self,'fp32_head_output',False),
                              "float32_matmul_precision": torch.get_float32_matmul_precision(),
                              "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32},
                "weights": _weight_metadata(Path(self.source_path)),
                "fp8_environment": fp8_environment_report() if config.get("quantization_config") else None}


@contextmanager
def gradient_projector(adapter):
    """Optional adapter optimization; generic/test adapters retain their API."""
    factory = getattr(adapter, 'scoped_gradient_projector', None)
    if factory is None:
        yield adapter.project
    else:
        with factory() as project:
            yield project


@contextmanager
def temporary_scoring_precision(adapter, dtype):
    """Promote frozen tensors for no-grad FD only; restore the exact tensor objects.

    LoRA masters are never replaced, merged or rounded. No training graph may
    span this context. Holding original storage makes restoration bit-exact,
    including exceptions, but the temporary FP32 storage costs real memory.
    """
    if dtype is None:
        yield
        return
    parameters=[p for p in adapter.model.parameters() if not p.requires_grad and p.is_floating_point()]
    originals=[(p,p.data) for p in parameters if p.dtype!=dtype]
    buffers=[]
    for module in adapter.model.modules():
        for name,b in module.named_buffers(recurse=False):
            if b is not None and b.is_floating_point() and b.dtype!=dtype:
                buffers.append((module,name,b))
    try:
        for p,_ in originals:p.data=p.data.to(dtype=dtype)
        for module,name,b in buffers:setattr(module,name,b.to(dtype=dtype))
        with torch.no_grad():yield
    finally:
        for p,original in originals:p.data=original
        for module,name,b in buffers:setattr(module,name,b)


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _weight_metadata(path):
    indices = sorted(path.glob("*.safetensors.index.json"))
    shards = sorted(path.glob("*.safetensors"))
    stats = [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
             for p in shards]
    return {"index_sha256": {p.name: _file_hash(p) for p in indices},
            "shard_stat_hash": fingerprint(stats), "shard_stats": stats,
            "binding": "index content SHA256 plus shard size/mtime_ns; NOT full shard content hashes"}


def attach_lora(model, targets, rank=8, alpha=16):
    model.requires_grad_(False)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Empty or duplicate LoRA manifest targets")
    modules = dict(model.named_modules())
    for name in targets:
        if not name.startswith("model.language_model.layers.") or not isinstance(modules.get(name), nn.Linear):
            raise ValueError(f"Manifest target is not an actual language linear: {name}")
    for name in targets:
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, FP32LoRALinear(modules[name], rank, alpha))
    # Includes tied embedding/head parameters (named_parameters removes duplicates).
    for name, p in model.named_parameters():
        is_lora = name.endswith((".lora_A", ".lora_B"))
        if p.requires_grad != is_lora or (is_lora and p.dtype != torch.float32):
            raise RuntimeError(f"Invalid frozen/FP32 parameter contract: {name}")
    if model.lm_head.weight.requires_grad or model.model.language_model.embed_tokens.weight.requires_grad:
        raise RuntimeError("Output head/embedding must be frozen, including tied weights")


def student_compute_dtype(precision):
    if precision not in ('bf16','fp32'):
        raise ValueError('student base_dtype must be bf16 or fp32')
    if precision=='fp32':
        # Shared by training and checkpoint evaluation through load_student.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        return torch.float32
    return torch.bfloat16


def load_student(path, manifest_path, *, device="cuda", gradient_checkpointing=True, rank=8, alpha=16,
                 base_dtype="bf16",fp32_head_output=False):
    from transformers import AutoConfig
    if torch.device(device).type != "cpu":
        require_gpu_guard()
    dtype=student_compute_dtype(base_dtype)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(config, "quantization_config", None):
        raise ValueError("Student base must be unquantized BF16")
    model = _conditional_class(config).from_pretrained(
        path, config=config, dtype=dtype, device_map=device,
        attn_implementation="eager", local_files_only=True)
    if dtype==torch.bfloat16:
        # vLLM's accepted Qwen GDN loader keeps A_log FP32. Native HF's global
        # dtype argument otherwise rounds the original FP32 checkpoint values.
        # Restore those tiny tensors from the ORIGINAL shards, not BF16 casts.
        from safetensors import safe_open
        names=dict(model.named_parameters())
        wanted={n for n in names if n.endswith('.linear_attn.A_log')}
        for shard in sorted(Path(path).glob('*.safetensors')):
            with safe_open(shard,framework='pt',device='cpu') as source:
                for name in wanted.intersection(source.keys()):
                    value=source.get_tensor(name)
                    if not value.is_floating_point():raise ValueError('Qwen A_log source must be floating point')
                    names[name].data=value.to(device=names[name].device,dtype=torch.float32)
                    wanted.remove(name)
        if wanted:raise ValueError('original Qwen A_log tensors missing')
    manifest = json.loads(Path(manifest_path).read_text())
    targets = manifest["target_modules"]
    attach_lora(model, targets, rank, alpha)
    adapter = QwenAdapter(model, trainable=True, gradient_checkpointing=gradient_checkpointing, source_path=path)
    adapter.fp32_head_output=fp32_head_output
    adapter.manifest_hash = fingerprint(manifest)
    adapter.lora_config = {"rank": rank, "alpha": alpha, "scaling": alpha / rank,
                           "dropout": 0.0, "master_dtype": "float32"}
    return adapter


def fp8_environment_report():
    import transformers
    return {"transformers": transformers.__version__, "torch": torch.__version__,
            "kernels_available": importlib.util.find_spec("kernels") is not None,
            "triton_available": importlib.util.find_spec("triton") is not None,
            "required_kernel": "kernels-community/finegrained-fp8",
            "kernel_revision": FP8_KERNEL_REVISION,
            "kernels_version": _kernels_version(),
            "disable_deepgemm_linear": os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "0")}


def _kernels_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("kernels")
    except PackageNotFoundError:
        return None


def prepare_fp8_kernel():
    """Resolve upstream v2 at an immutable SHA, before allocating either model.

    Supports predownloaded HF_HUB_CACHE plus HF_HUB_OFFLINE=1. No framework
    patch or custom kernel: register the pinned upstream kernel with the native
    Transformers dispatcher, avoiding a network-only v2 branch lookup.
    """
    if not fp8_environment_report()["kernels_available"]:
        raise RuntimeError("FP8 requires isolated compatible kernels>=0.12,<0.13; no silent BF16 fallback")
    from kernels import get_kernel, get_local_kernel
    from transformers.integrations import hub_kernels
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = cache / "models--kernels-community--finegrained-fp8" / "snapshots" / FP8_KERNEL_REVISION
    if snapshot.is_dir():
        source = snapshot / "build/torch-cuda"
        hashes = {str(p.relative_to(source)): _file_hash(p) for p in sorted(source.rglob("*"))
                  if p.is_file() and (p.suffix == ".py" or p.name == "metadata.json")}
        if fingerprint(hashes) != FP8_KERNEL_SOURCE_HASH:
            raise RuntimeError("Pinned FP8 kernel source hash mismatch")
        kernel = get_local_kernel(snapshot, "cq_pinned_finegrained_fp8")
    else:
        kernel = get_kernel("kernels-community/finegrained-fp8", revision=FP8_KERNEL_REVISION)
    missing = [name for name in ("matmul", "matmul_batched", "matmul_grouped")
               if not callable(getattr(kernel, name, None))]
    if missing:
        raise RuntimeError(f"Pinned FP8 kernel lacks required symbols: {missing}")
    hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = kernel
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
    load_finegrained_fp8_kernel()
    return {"revision": FP8_KERNEL_REVISION, "module": kernel.__name__,
            "kernels_version": _kernels_version()}


def load_teacher(path, *, device="cuda"):
    from transformers import AutoConfig
    require_gpu_guard()
    if torch.device(device).type != "cuda":
        raise RuntimeError("FP8 Teacher requires CUDA; Transformers CPU fallback dequantization is forbidden")
    report = fp8_environment_report()
    if not report["kernels_available"]:
        raise RuntimeError(f"FP8 Teacher kernel dependency missing; no conversion/install permitted: {report}")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(torch.device(device)) < (8, 9):
        raise RuntimeError("Unsupported CUDA device would trigger forbidden BF16 dequantization")
    prepare_fp8_kernel()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    qc = getattr(config, "quantization_config", {})
    if qc.get("quant_method") != "fp8" or qc.get("dequantize", False):
        raise ValueError("Teacher must retain checkpoint FP8 with dequantize=False")
    # Use checkpoint quantizer defaults; never replace/remove quantization_config.
    model = _conditional_class(config).from_pretrained(
        path, config=config, device_map=device, local_files_only=True, attn_implementation="eager")
    actual = model.hf_quantizer.quantization_config
    if actual.dequantize or not any(p.dtype == torch.float8_e4m3fn for p in model.parameters()):
        raise RuntimeError("Teacher silently dequantized instead of retaining FP8 checkpoint")
    model.requires_grad_(False)
    return QwenAdapter(model, source_path=path)


def tokenizer_metadata(tokenizer, *, thinking=False):
    specials = {k: str(v) if not isinstance(v, list) else [str(x) for x in v]
                for k, v in tokenizer.special_tokens_map.items()}
    special_ids = {k: ([tokenizer.convert_tokens_to_ids(str(x)) for x in v] if isinstance(v, list)
                       else tokenizer.convert_tokens_to_ids(str(v))) for k, v in specials.items()}
    return {"path": tokenizer.name_or_path, "revision": tokenizer.init_kwargs.get("_commit_hash"),
            "vocab_hash": fingerprint(tokenizer.get_vocab()), "special_tokens_hash": fingerprint(specials),
            "special_ids_hash": fingerprint(special_ids), "template_hash": fingerprint(tokenizer.chat_template),
            "thinking_hash": fingerprint({"enable_thinking": thinking})}


def validate_alignment(student, teacher, student_tokenizer, teacher_tokenizer, *, thinking=False):
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher/student token string-to-ID mapping differs")
    s = tokenizer_metadata(student_tokenizer, thinking=thinking)
    t = tokenizer_metadata(teacher_tokenizer, thinking=thinking)
    for key in ("special_tokens_hash", "special_ids_hash"):
        if s[key] != t[key]:
            raise ValueError(f"Teacher/student {key} differs")
    sv, tv = student.model.lm_head.out_features, teacher.model.lm_head.out_features
    if sv != tv or max(student_tokenizer.get_vocab().values()) >= sv:
        raise ValueError("Output vocabulary dimensions differ or cannot represent tokenizer IDs")
    result = {"student": student.metadata(), "teacher": teacher.metadata(),
              "student_tokenizer": s, "teacher_tokenizer": t, "output_vocab": sv,
              "prompt_policy": "student_template_once_shared_ids", "thinking": thinking,
              "lora_manifest_hash": getattr(student, "manifest_hash", None)}
    result["alignment_hash"] = fingerprint(result)
    return result


def prompt_ids(tokenizer, prompt, *, thinking=False):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=thinking, return_dict=False)


def check_logits_equivalence(adapter, batch, *, atol=0.02, rtol=0.02):
    adapter.set_scoring_mode()
    with torch.no_grad():
        standard = adapter.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 use_cache=False, past_key_values=None).logits
        train_hidden = adapter.last_hidden(batch, with_grad=False)
        projected = adapter.project(train_hidden)
        torch.testing.assert_close(projected, standard, atol=atol, rtol=rtol)
        adapter.model.eval()
        eval_hidden = adapter.language_model(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"],
                                             use_cache=False).last_hidden_state
        adapter.set_scoring_mode()
        torch.testing.assert_close(train_hidden, eval_hidden, atol=atol, rtol=rtol)
    return {"logits_max_abs": float((projected.float() - standard.float()).abs().max()),
            "mode_hidden_max_abs": float((train_hidden.float() - eval_hidden.float()).abs().max())}
