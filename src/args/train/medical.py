import transformers
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

@dataclass
class ModelArguments:
  
    lang_encoder_path: Optional[str] = field(
        default="./GPT-2")
    tokenizer_path: str = field(default="./GPT-2",
                                metadata={"help": "Path to the tokenizer data."})
    pretrained_visual_encoder: Optional[str] = field(
      
        default="./RadFM_vit3d.pth")
    pretrained_adapter: Optional[str] = field(
       
        default="./RadFM_perceiver_fc.pth")

@dataclass
class DataArguments:
    data_folder: Optional[str] = field(default='./data/global')
    mask_folder: Optional[str] = field(default='./data/region')
    report_file: Optional[str] = field(default='./data/reports.csv')
    monai_cache_dir: Optional[str] = field(default='./data/cache')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(
        default="./outputs")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pin_memory: bool = field(default=True)