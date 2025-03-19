import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig

model_name_or_path = "/media/caiyunfeng/Expansion/cuixiaoteng/checkpoints/llava_factory/sft-TinyLLaVA-Phi-2-SigLIP-3.1B-lora-direct-geo3k-0820-point-ep8"

merged_model_save_path = "/media/caiyunfeng/Expansion/cuixiaoteng/checkpoints/llava_factory/sft-TinyLLaVA-Phi-2-SigLIP-3.1B-lora-direct-geo3k-0820-point-ep8-merged"


def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp


if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
    model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
    model = TinyLlavaForConditionalGeneration(model_config)
    language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
    language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
    model.language_model.load_state_dict(language_model_ckp)
    vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
    vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
    model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
    connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
    connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
    model.connector.load_state_dict(connector_ckp)
    model.to(torch.float16)
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_name_or_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    model.save_pretrained(merged_model_save_path)