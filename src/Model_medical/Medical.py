from torch import nn
from transformers.models.llama import LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from .my_embedding_layerd import MyEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import tqdm.auto as tqdm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


class Medical(nn.Module):
    def __init__(
        self,
        text_tokenizer_path,
        lang_model_path,
        pretrained_visual_encoder,
        pretrained_adapter,
        max_region_size=10,
        max_img_size=1,
        image_num=32,
    ):
        super(Medical, self).__init__()

        self.image_padding_tokens = []
        self.text_tokenizer = LlamaTokenizer.from_pretrained(
            text_tokenizer_path,
        )

        special_token = {
            "additional_special_tokens": [
                "<image>",
                "</image>",
                "<region>",
                "</region>",
            ]
        }

        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(image_token)
            self.image_padding_tokens.append(image_padding_token)

        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(image_num + 1):
                region_token = "<region" + str(i * image_num + j) + ">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(region_token)
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(special_token)

        self.text_tokenizer.pad_token_id = 0
        self.text_tokenizer.bos_token_id = 1
        self.text_tokenizer.eos_token_id = 2

        self.lang_model = LlamaForCausalLM.from_pretrained(
            lang_model_path,
        )
        self.lang_model = self.lang_model.half()

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.lang_model = get_peft_model(
            self.lang_model,
            peft_config,
        )
        self.lang_model.print_trainable_parameters()

        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()

        self.embedding_layer = MyEmbedding(
            pretrained_visual_encoder,
            pretrained_adapter,
        )
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight

        self.loss_function = nn.CrossEntropyLoss()

        self.hidden_dim = 4096
        self.voc_size = 32000

    def forward(
        self,
        lang_x,
        vision_x,
        mask_x,
        region2area,
        attention_mask,
        labels,
    ):
        if labels.shape == lang_x.shape:

            input_embedding = self.embedding_layer(
                vision_x,
                mask_x,
                lang_x,
                region2area,
            )

            output = self.lang_model(
                inputs_embeds=input_embedding,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits = output["logits"][..., :-1, :].contiguous().detach()

            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)

            labels = labels[..., 1:].contiguous()

            acc = torch.sum(
                torch.all(
                    torch.logical_or(
                        predictions == labels,
                        labels == -100,
                    ),
                    dim=-1,
                )
            )

            accuracy = acc / total

            if torch.distributed.get_rank() == 0:
                print("lm_loss:", output["loss"].item())

            return {
                "logits": accuracy,
                "loss": output["loss"],
            }

    def generate(
        self,
        lang_x,
        vision_x,
        mask_x,
        region2area,
    ):
        with torch.no_grad():

            input_embedding = self.embedding_layer(
                vision_x,
                mask_x,
                lang_x,
                region2area,
            )

            input_embedding = input_embedding.to(
                next(self.lang_model.parameters()).dtype
            )

            generation = self.lang_model.generate(
                inputs_embeds=input_embedding,
                max_new_tokens=500,
                top_k=30,
            )

            report = self.text_tokenizer.batch_decode(
                generation,
                skip_special_tokens=True,
            )

        return report