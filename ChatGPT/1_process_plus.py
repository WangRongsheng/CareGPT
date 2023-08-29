import json
from tqdm import tqdm

def transform_json(input_file_path, output_file_path):
    with open(input_file_path, encoding='utf-8') as file:
        data = json.load(file)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for i in tqdm(range(len(data))):
            messages = []
            messages.append({"role": "system", "content": "You are an experienced doctor."}) # prompt
            user_message = {"role": "user", "content": data[i]["instruction"]} # 修改instruction / input
            assistant_message = {"role": "assistant", "content": data[i]["output"]} # 修改output
            messages.extend([user_message, assistant_message])
            result = {"messages": messages}
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')

input_file_path = 'sft-20k.json' # 请替换为您的输入JSONL文件路径
output_file_path = 'output-plus.json' # 请替换为您想要保存的输出JSONL文件路径
transform_json(input_file_path, output_file_path)