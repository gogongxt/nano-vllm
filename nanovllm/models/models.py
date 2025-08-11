from .llama import LlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM

model_dict = {
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}
