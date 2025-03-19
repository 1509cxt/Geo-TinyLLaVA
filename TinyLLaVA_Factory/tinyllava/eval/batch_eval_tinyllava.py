import argparse
import re
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json
import os
import re

import torch
from transformers import PreTrainedModel

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model(args.model_path)
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor
    
    with open(args.eval_data_file, 'r') as file:
        eval_data_json = json.load(file)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.cuda()

    eval_output_list=[]
    for eval_case in tqdm(eval_data_json):
        qs = eval_case["conversations"][0]['value']
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        msg = Message()
        msg.add_message(qs)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()
            
        image_file = os.path.join(args.image_folder_dir, eval_case["image"])
        image_files = [image_file]
        images = load_images(image_files)[0]
        images_tensor = image_processor(images)
        images_tensor = images_tensor.unsqueeze(0).half().cuda()

        

        stop_str = text_processor.template.separator.apply()[1]
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        #eval_output_list.append(outputs)

        pattern=r"\/(\d+)\/"
        match = re.search(pattern, image_file)
        id = match.group(1)
        with open(os.path.join(args.eval_output_dir, id+".txt"), "w") as text_file:
            text_file.write(outputs)
        
        print(id, ": ", outputs)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=PreTrainedModel, default=None)
    parser.add_argument("--eval-data-file", type=str, required=True)
    parser.add_argument("--image-folder-dir", type=str, required=True)
    parser.add_argument("--eval-output-dir", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)