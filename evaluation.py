import os
import json
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, MllamaForConditionalGeneration, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration, CLIPImageProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from qwen_vl_utils import process_vision_info


device = 'cuda'
torch.set_default_device(device)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_prompt_indefinite(data):
    question = data['question']
    options = data['options']
    length = len(options)
    choices = ''
    if length == 2:
        choices = 'A, B'
    elif length == 3:
        choices = 'A, B, C'
    elif length == 4:
        choices = 'A, B, C, D'
    elif length == 5:
        choices = 'A, B, C, D, E'
    system_prompt = f"Give an indefinite multiple choice question with several options, you need to select all the options that can answer the question. \
        The output should only contains the option index and strictly follow this format: {choices}, and don't contains any other contents!!\n"
    prompt =  system_prompt + "Question: " + question+'\n' + 'Options: \n'
    if length == 2:
        caption1 = options['A']
        caption2 = options['B']
        prompt += f"(A) {caption1}\n(B) {caption2}\n"
    elif length == 3:
        caption1 = options['A']
        caption2 = options['B']
        caption3 = options['C']
        prompt += f"(A) {caption1}\n(B) {caption2}\n(C) {caption3}\n"
    elif length == 4:
        caption1 = options['A']
        caption2 = options['B']
        caption3 = options['C']
        caption4 = options['D']
        prompt += f"(A) {caption1}\n(B) {caption2}\n(C) {caption3}\n(D) {caption4}\n"
    elif length == 5:
        caption1 = options['A']
        caption2 = options['B']
        caption3 = options['C']
        caption4 = options['D']
        caption5 = options['E']
        prompt += f"(A) {caption1}\n(B) {caption2}\n(C) {caption3}\n(D) {caption4}\n(E) {caption5}\n"
    return prompt

def inference(args):
    inp_path = os.path.join(args.datasetpath,'images')
    with open(os.path.join(args.datasetpath,'questions.json'),'r') as f:
        data = json.load(f)
    length = len(data)
    img_li = sorted(os.listdir(inp_path),key=lambda x:int(x.split('.')[0]))

    if 'Bunny' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,device_map='auto',trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    elif 'InternVL2' in args.model:
        model = AutoModel.from_pretrained(args.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    elif 'InternVL-Chat-V1-5' in args.model:
        model = AutoModel.from_pretrained(args.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    elif 'InternVL-Chat-V1-2' in args.model:
        model = AutoModel.from_pretrained(args.model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=True,trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(args.model)
    elif 'internlm-xcomposer' in args.model:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    elif 'Llama-3.2' in args.model:
        model = MllamaForConditionalGeneration.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model)
    elif 'llava-v1.6' in args.model:
        model = LlavaNextForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
        processor = LlavaNextProcessor.from_pretrained(args.model)
    elif 'llava-1.5' in args.model:
        model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True, ).to("cuda")
        processor = AutoProcessor.from_pretrained(args.model)
    elif 'MiniCPM' in args.model:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True,attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    elif 'Phi-3-vision-128k-instruct' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True) 
    elif 'Phi-3.5-vision-instruct' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model,device_map="cuda",trust_remote_code=True,torch_dtype="auto",_attn_implementation='flash_attention_2')
        processor = AutoProcessor.from_pretrained(args.model,trust_remote_code=True, num_crops=4) 
    elif 'Qwen-VL' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    elif 'Qwen2-VL' in args.model:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model)

    result = {}
    result_name = args.model.split('/')[-1] + '.json'
    for i in tqdm(range(length)):
        temp_prompt = get_prompt_indefinite(data[i])
        image_path = os.path.join(inp_path,img_li[i])
        if 'Bunny' in args.model:
            text = f"USER: <image>\n{temp_prompt} ASSISTANT:"
            text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)
            image = Image.open(image_path)
            image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
            output_ids = model.generate(input_ids,images=image_tensor,max_new_tokens=1024,use_cache=False,repetition_penalty=1.0)[0]
            result[img_li[i]] = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        elif 'InternVL2' in args.model:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            result[img_li[i]] = model.chat(tokenizer, pixel_values, temp_prompt, generation_config)
        elif 'InternVL-Chat-V1-5' in args.model:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            result[img_li[i]] = model.chat(tokenizer, pixel_values, '<image>\n'+temp_prompt, generation_config)
        elif 'InternVL-Chat-V1-2' in args.model:
            pixel_values = image_processor(images=Image.open(image_path).resize((448,448)), return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            result[img_li[i]] = model.chat(tokenizer, pixel_values, '<image>\n'+temp_prompt, generation_config)
        elif 'internlm-xcomposer2d5' in args.model:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result[img_li[i]], _ = model.chat(tokenizer, temp_prompt, [image_path], do_sample=False, num_beams=1, use_meta=True)
        elif 'internlm-xcomposer2' in args.model:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result[img_li[i]], _ = model.chat(tokenizer, '<ImageHere>'+temp_prompt, [image_path], do_sample=False, num_beams=1, use_meta=True)
        elif 'Llama-3.2' in args.model:
            messages = [{"role": "user", "content": [{"type": "image"},{"type": "text", "text": temp_prompt}]}]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(Image.open(image_path), input_text, return_tensors="pt").to(model.device)
            output = processor.decode(model.generate(**inputs, max_new_tokens=1024)[0])
            result[img_li[i]] = output.split('<|end_header_id|>\n\n')[-1].replace('<|eot_id|>','')
        elif 'llava-v1.6' in args.model and 'mistral' in args.model:
            conversation = [{"role": "user","content": [{"type": "text", "text": f"{temp_prompt}"},{"type": "image"},],},]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=Image.open(image_path), text=prompt, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=1024)
            result[img_li[i]] = processor.decode(output[0], skip_special_tokens=True).split('[/INST] ')[-1]
        elif 'llava-v1.6' in args.model:
            conversation = [{"role": "user","content": [{"type": "text", "text": f"{temp_prompt}"},{"type": "image"},],},]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=Image.open(image_path), text=prompt, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=1024)
            result[img_li[i]] = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[-1]
        elif 'llava-1.5' in args.model:
            conversation = [{"role": "user","content": [{"type": "text", "text": f"{temp_prompt}"},{"type": "image"},],},]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=Image.open(image_path), text=prompt, return_tensors='pt').to(0,torch.float16)
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            result[img_li[i]] = processor.decode(output[0][2:], skip_special_tokens=True).split('ASSISTANT: ')[-1]
        elif 'MiniCPM' in args.model:
            image = Image.open(image_path)
            msgs = [{'role': 'user', 'content': [image, question]}]
            result[img_li[i]] = model.chat(image=None,msgs=msgs,tokenizer=tokenizer)
        elif 'Phi-3' in args.model:
            messages = [{"role": "user", "content": f"<|image_1|>\n{temp_prompt}"}] 
            image = Image.open(image_path)
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 
            generation_args = {"max_new_tokens": 1024, "temperature": 0.0, "do_sample": False, } 
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            result[img_li[i]] = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        elif 'Qwen-VL' in args.model:
            query = tokenizer.from_list_format([{'image': image_path}, {'text': temp_prompt},])
            result[img_li[i]], _ = model.chat(tokenizer, query=query, history=None)
        elif 'Qwen2-VL' in args.model:
            messages = [{"role": "user","content": [{"type": "image", "image": f"file://{image_path}"},{"type": "text", "text": f"{temp_prompt}"},],}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor( text=[text],images=image_inputs,videos=video_inputs, padding=True,return_tensors="pt",).to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            result[img_li[i]] = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        with open(result_name, 'w') as f:
            json.dump(result,f,indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='huggingface model link')
    parser.add_argument('--datasetpath', type=str, help='local address of MMComposition')
    args = parser.parse_args()
    inference(args)
