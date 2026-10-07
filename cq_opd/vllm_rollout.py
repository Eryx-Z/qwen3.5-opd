"""CQ bridge to the accepted baseline vLLM loader and verl tensor-LoRA sync.

Only inference is delegated. Teacher, full-vocabulary KL, FD and optimizer
remain in their existing owners. No baseline/framework files are patched.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from numbers import Integral
import os
import time
import torch

from .model_adapter import require_gpu_guard
from .rollout import Rollout
from .lora_integrity import verify_receiver


def baseline_engine_kwargs(model,*,max_num_seqs=16):
    if isinstance(max_num_seqs,bool) or not isinstance(max_num_seqs,Integral) or max_num_seqs<1:
        raise ValueError('max_num_seqs must be a positive integer')
    # Same verified baseline/eval settings; eager and no prefix cache preserve
    # safe parameter mutation. BF16 is required by the installed GDN kernels.
    return dict(model=model,dtype='bfloat16',tensor_parallel_size=1,trust_remote_code=False,
        enable_lora=True,max_lora_rank=8,max_loras=1,lora_dtype='bfloat16',max_model_len=2561,
        max_num_seqs=int(max_num_seqs),max_num_batched_tokens=4096,kv_cache_memory_bytes=4294967296,
        enforce_eager=True,enable_sleep_mode=False,seed=42,generation_config='vllm',
        enable_prefix_caching=False,max_logprobs=-1,gpu_memory_utilization=.4,
        limit_mm_per_prompt={'image':0,'video':0},skip_mm_profiling=True,
        worker_extension_cls='cq_opd.vllm_integrity_worker.IntegrityWorker')


def tensor_lora_payload(adapter):
    """Use baseline TensorLoRARequest's PEFT names; never mutate master weights."""
    tensors={};digest=hashlib.sha256();targets=set()
    for name,p in adapter.lora_named_parameters():
        if not name.startswith('model.language_model.') or not name.endswith(('.lora_A','.lora_B')):
            raise ValueError('unverified language-only LoRA tensor name')
        if p.dtype!=torch.float32 or not torch.isfinite(p).all():
            raise ValueError('LoRA master must be finite FP32')
        t=p.detach().cpu().contiguous().clone()
        digest.update(name.encode());digest.update(t.numpy().tobytes())
        tensors['base_model.model.'+name+'.weight']=t
        targets.add(name.rsplit('.',2)[1])
    if not tensors or len(tensors)%2:raise ValueError('missing paired LoRA tensors')
    c=adapter.lora_config
    config=dict(peft_type='LORA',task_type='CAUSAL_LM',r=c['rank'],lora_alpha=c['alpha'],
                lora_dropout=0.,bias='none',target_modules=sorted(targets),inference_mode=True)
    return tensors,config,digest.hexdigest()


def baseline_transfer(engine,tensors,config):
    """The existing baseline bucketed CUDA-IPC transport, not JSON tensor RPC."""
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
    handles=engine.collective_rpc('_get_zmq_handle')
    if len(handles)!=1:raise RuntimeError('CQ baseline bridge requires single GPU')
    with ThreadPoolExecutor(max_workers=1) as pool:
        receiver=pool.submit(engine.collective_rpc,'update_weights_from_ipc',kwargs=dict(
            peft_config=config,base_sync_done=True,use_shm=False))
        sender=BucketedWeightSender(zmq_handle=handles[0],bucket_size_mb=64,use_shm=False)
        asyncio.run(sender.async_send_weights(list(tensors.items())))
        return receiver.result(timeout=60)


class BaselineVLLMRollout:
    def __init__(self,adapter,tokenizer,*,engine=None,sampling_cls=None,parity_scope='sampled',max_num_seqs=16):
        engine_kwargs=baseline_engine_kwargs(adapter.source_path,max_num_seqs=max_num_seqs)
        require_gpu_guard()
        if adapter.device.type!='cuda' and engine is None:raise ValueError('vLLM requires guarded CUDA')
        if adapter.model.get_input_embeddings().weight.dtype!=torch.bfloat16:
            raise ValueError('installed baseline vLLM requires BF16 Student; no silent dtype fallback')
        if engine is None:
            os.environ.setdefault('VERL_RAY_JOB_ID',os.environ['VERL_OPD_GUARD_RUN'])
            from vllm import LLM,SamplingParams
            engine=LLM(**engine_kwargs);sampling_cls=SamplingParams
        self.engine=engine;self.sampling_cls=sampling_cls;self.adapter=adapter;self.tokenizer=tokenizer
        if parity_scope not in ('sampled','full'):raise ValueError('invalid policy parity scope')
        self.parity_scope=parity_scope
        self.policy_hash=None;self.ready=False;self.sync_seconds=0.;self.policy_checks=[]
        other=engine.get_tokenizer()
        if tokenizer.get_vocab()!=other.get_vocab() or tokenizer.all_special_ids!=other.all_special_ids:
            raise ValueError('vLLM tokenizer or special-token mismatch')

    def sync_policy(self):
        from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
        tensors,config,sha=tensor_lora_payload(self.adapter)
        if self.ready and sha==self.policy_hash:return False
        self.ready=False;started=time.monotonic()
        # EXACT existing baseline's remove -> TensorLoRARequest -> add flow.
        self.engine.llm_engine.remove_lora(VLLM_LORA_INT_ID)
        result=baseline_transfer(self.engine,tensors,config)
        if result is None:raise RuntimeError('baseline weight sync returned no acknowledgements')
        if VLLM_LORA_INT_ID not in self.engine.llm_engine.list_loras():
            raise RuntimeError('current LoRA missing after baseline sync')
        if not self.engine.reset_prefix_cache():raise RuntimeError('old inference cache not cleared')
        received=self.engine.collective_rpc('cq_lora_fingerprints')
        if len(received)!=1:raise RuntimeError('expected single receiver integrity report')
        self.integrity_check=verify_receiver(received[0],tensors,config)
        self.policy_hash=sha;self.sync_seconds+=time.monotonic()-started
        # Exact receiver integrity is mandatory; finite probability drift is diagnostic.
        self.validate_policy()
        self.ready=True
        return True

    def _request(self):
        from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID,VLLM_LORA_NAME,VLLM_LORA_PATH
        from vllm.lora.request import LoRARequest
        return LoRARequest(VLLM_LORA_NAME,VLLM_LORA_INT_ID,VLLM_LORA_PATH)

    def validate_policy(self):
        ids=self.tokenizer.apply_chat_template([{'role':'user','content':'What is 2 + 3?'}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        params=self.sampling_cls(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,n=1,seed=42,
                                logprobs=-1 if self.parity_scope=='full' else 0,detokenize=False)
        outputs=self.engine.generate([{'prompt_token_ids':ids}],params,lora_request=self._request(),use_tqdm=False)
        entries=outputs[0].outputs[0].logprobs[0]
        x=torch.tensor([ids],device=self.adapter.device,dtype=torch.long)
        with torch.no_grad():
            h=self.adapter.last_hidden(dict(input_ids=x,attention_mask=torch.ones_like(x,dtype=torch.bool)),False)
            native=torch.log_softmax(self.adapter.project(h[:,-1]).float(),-1)[0].cpu()
        vocab_size=native.numel()
        if self.parity_scope=='full':
            if len(entries)!=vocab_size or set(entries)!=set(range(vocab_size)):
                raise RuntimeError('policy parity requires entire actual vocabulary')
            checked=list(range(vocab_size))
        else:
            checked=list(outputs[0].outputs[0].token_ids)
            if len(checked)!=1 or checked[0] not in entries:
                raise RuntimeError('policy parity missing sampled token')
            native=native[checked]
        accelerated=torch.tensor([entries[i].logprob for i in checked])
        if not torch.isfinite(accelerated).all() or not torch.isfinite(native).all():
            raise RuntimeError('nonfinite rollout/native policy')
        within=bool(torch.allclose(accelerated,native,atol=.02,rtol=.02))
        self.policy_checks.append(dict(policy_hash=self.policy_hash,passed=within,atol=.02,rtol=.02,
            enforcement='diagnostic_only',within_reference_tolerance=within,
            receiver_integrity=getattr(self,'integrity_check',None),
            mean_abs=float((accelerated-native).abs().mean()),
            scope=self.parity_scope,checked_tokens=len(checked),
            max_abs=float((accelerated-native).abs().max()),vocab_size=vocab_size))

    def generate(self,prompts,cap,seed,ids):
        if not prompts or cap<1 or len(prompts)!=len(ids):raise ValueError('invalid rollout batch')
        self.sync_policy()
        if not self.ready:raise RuntimeError('unverified or stale rollout policy')
        sampling=[self.sampling_cls(temperature=1.,top_p=1.,top_k=-1,repetition_penalty=1.,
                    max_tokens=cap,n=1,seed=(seed+i)%2**31,detokenize=False,logprobs=0) for i in range(len(prompts))]
        outputs=self.engine.generate([{'prompt_token_ids':p} for p in prompts],sampling,
                                    lora_request=self._request(),use_tqdm=False)
        if len(outputs)!=len(prompts):raise RuntimeError('vLLM lost rollout rows')
        eos=self.adapter.model.config.text_config.eos_token_id
        eos=set(eos if isinstance(eos,(list,tuple)) else [eos])
        records=[]
        for i,(p,output) in enumerate(zip(prompts,outputs)):
            if list(output.prompt_token_ids)!=p or len(output.outputs)!=1:raise RuntimeError('vLLM prefix mismatch')
            response=output.outputs[0];tokens=list(response.token_ids)
            if not tokens or len(tokens)>cap or response.finish_reason not in ('stop','length'):
                raise RuntimeError('invalid or unfinished vLLM response')
            # Native engine returns EOS before stopping. Refuse interior EOS or
            # fake suffixes; do not synthesize EOS if it stops without one.
            if any(t in eos for t in tokens[:-1]):raise RuntimeError('tokens after actual EOS')
            entries=response.logprobs
            if entries is None or len(entries)!=len(tokens) or any(t not in e for t,e in zip(tokens,entries)):
                raise RuntimeError('missing sampled rollout logprobs')
            logps=[e[t].logprob for t,e in zip(tokens,entries)]
            if not torch.isfinite(torch.tensor(logps)).all():raise RuntimeError('nonfinite sampled rollout')
            truncated=len(tokens)==cap and tokens[-1] not in eos
            records.append(Rollout(p+tokens,len(p),tokens,
                self.tokenizer.decode(tokens,skip_special_tokens=True),str(ids[i]),truncated,
                (seed+i)%2**31,generation_batch_size=len(prompts),sample_index=i,
                policy_sha256=self.policy_hash,sampled_log_probs=logps))
        return records

    def close(self):
        if self.engine is not None:
            self.engine.llm_engine.engine_core.shutdown()
            self.engine=None
