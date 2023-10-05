import os
import torch
import platform
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers.generation.utils import GenerationConfig
from transformers.generation import GenerationConfig
import json
from tqdm import tqdm
import re

def init_model():
    print("init model ...")
    '''
    model = AutoModelForCausalLM.from_pretrained(
        "./exportchatml",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "./exportchatml"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "./exportchatml",
        use_fast=False,
        trust_remote_code=True
    )
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        "./exportchatml", trust_remote_code=True, resume_download=True,
    )

    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        "./exportchatml",
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    '''
    config = GenerationConfig.from_pretrained(
        "./exportchatml", trust_remote_code=True, resume_download=True,
    )
    '''
    config = GenerationConfig(chat_format='chatml', eos_token_id=151643, pad_token_id=151643, max_window_size=6144, max_new_tokens=512, do_sample=True, top_k=0, top_p=0.5)
    
    return model, tokenizer, config

def main():
    model, tokenizer, config = init_model()
    #tokenizer = AutoTokenizer.from_pretrained("./exportchatml", trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained("./exportchatml", trust_remote_code=True)
    #model = model.eval()
    
    with open('CMB-test-choice-question-merge.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    for i in tqdm(range(len(data))):
        messages = []
        
        test_id = data[i]['id']
        exam_type = data[i]['exam_type']
        exam_class = data[i]['exam_class']
        question_type = data[i]['question_type']
        question = data[i]['question']
        option_str = 'A. ' + data[i]['option']['A'] + '\nB. ' + data[i]['option']['B']+ '\nC. ' + data[i]['option']['C']+ '\nD. ' + data[i]['option']['D']
        
        #prompt = f'以下是中国{exam_type}中{exam_class}考试的一道{question_type}，不需要做任何分析和解释，直接给出正确选项。\n{question}\n{option_str}'
        prompt = f'以下是一道{question_type}，不需要做任何分析解释，直接给出正确选项：\n{question}\n{option_str}'
        #messages.append({"role": "user", "content": prompt})
        #history = ''
        #response = model.chat(tokenizer, messages,history)
        #print(response)
        #print(type(response))
        '''
        generation_config = GenerationConfig(max_new_tokens=1024)
        
        text = 'User: '+prompt+'<|endoftext|>\n Assistant: '
        inputs = tokenizer.encode(text, return_tensors="pt").to('cpu')
        outputs = model.generate(inputs, generation_config=generation_config)
        output = tokenizer.decode(outputs[0])
        response = output.replace(inputs, '')
        print(response)
        '''
        history = []
        response, history = model.chat(tokenizer, prompt, history=history, generation_config=config)
        print(response)
        matches = re.findall("[ABCDE]", response)
        print(matches)
        final_result = "".join(matches)
    
        info = {
                "id": test_id,
                "model_answer": final_result
            }
        results.append(info)
        
        history.clear()

    with open('output.json', 'w', encoding="utf-8") as f1:
        json.dump(results, f1, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
