import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="./pretrained_models/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="./pretrained_models/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
        default="./pretrained_models/RadFM_vit3d.pth")
    pretrained_adapter: Optional[str] = field(
        default="./pretrained_models/RadFM_perceiver_fc.pth")
    ckpt_path: Optional[str] = field(
        default="./pytorch_model.bin")

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='./data/global')
    mask_folder: Optional[str] = field(default='./data/region')
    report_file: Optional[str] = field(default='./data/reports.csv')
    monai_cache_dir: Optional[str] = field(default='./data/cache')
    result_path: Optional[str] = field(default='./results/reports.csv')