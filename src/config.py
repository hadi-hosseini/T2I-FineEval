from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    api_key: str = "API_KEY"
    api_model_name: str = "gpt-4"
    api_base_url: str = "https://api.avalai.ir/v1"  # use OPENAI instead.
    api_temperature: float = 0.2
    api_max_tokens: int = 1024
    api_top_p: float = 1
    api_chat_mode: bool = True  # True: GPT-4 False: GPT-3.5
    create_pdf: bool = False
    create_heatmap: bool = False
    fine_grained_coef: int = 1  # fine-grained coefficient
    coarse_grained_coef: int = 1  # coarse-grained coefficient
    colors: List[str] = field(default_factory=lambda: [
                              "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"])
    prompt_decomposition_path: str = Path(
        "/home/mmd/Diffusion-Eval/comtie/t2i_evaluation/inputs/prompt_decomposition.txt")
    disjoint_question_generation: str = Path(
        "/home/mmd/Diffusion-Eval/comtie/t2i_evaluation/inputs/disjoint_question_generation.txt")
    decompose_system_prompt: str = "You are ChatGPT, a model which breaksdown complex captions into a decomposable format. Answer as concisely as possible."
    disjoin_system_prompt: str = "You are ChatGPT, a model which breaksdown captions of varying complexity into simpler assertions/questions. Answer as concisely as possible."
    extend_threshold: float = 0.2
