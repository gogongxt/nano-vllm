from .llama import LlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3 import Qwen3ForCausalLM

model_dict = {
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}
