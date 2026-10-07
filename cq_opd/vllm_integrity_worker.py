"""Project-owned extension of unchanged baseline IPC worker."""
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension, VLLM_LORA_INT_ID
from .lora_integrity import fingerprint


class IntegrityWorker(vLLMColocateWorkerExtension):
    def cq_lora_fingerprints(self):
        adapter=self.model_runner.lora_manager._adapter_manager.get_adapter(VLLM_LORA_INT_ID)
        if adapter is None:raise RuntimeError('missing current receiver LoRA')
        result=[]
        for name,lora in adapter.loras.items():
            for kind in ('lora_a','lora_b'):
                values=getattr(lora,kind)
                if not isinstance(values,(tuple,list)):values=[values]
                for index,tensor in enumerate(values):
                    if tensor is not None:
                        result.append(dict(name=name,kind=kind,index=index,**fingerprint(tensor)))
        return result
