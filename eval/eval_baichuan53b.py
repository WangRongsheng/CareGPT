import requests
import json
import time
import hashlib
from tqdm import tqdm
import re

def calculate_md5(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    encrypted = md5.hexdigest()
    return encrypted

def do_request(text):
    url = "https://api.baichuan-ai.com/v1/chat"
    api_key = "apikey"
    secret_key = "screctkey"

    data = {
        "model": "Baichuan2-53B",
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ]
    }

    json_data = json.dumps(data)
    time_stamp = int(time.time())
    signature = calculate_md5(secret_key + json_data + str(time_stamp))

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
        "X-BC-Request-Id": "your requestId",
        "X-BC-Timestamp": str(time_stamp),
        "X-BC-Signature": signature,
        "X-BC-Sign-Algo": "MD5",
    }
    
    try:
        response = requests.post(url, data=json_data, headers=headers)
        return str(response.json()['data']['messages'][0]['content'])
    except:
        return ''
    

def main():
    with open('CMB-test-choice-question-merge.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    c = 0
    for i in tqdm(range(len(data))):
        
        test_id = data[i]['id']
        exam_type = data[i]['exam_type']
        exam_class = data[i]['exam_class']
        question_type = data[i]['question_type']
        question = data[i]['question']
        option_str = 'A. ' + data[i]['option']['A'] + '\nB. ' + data[i]['option']['B']+ '\nC. ' + data[i]['option']['C']+ '\nD. ' + data[i]['option']['D']
        
        prompt = f'以下是中国{exam_type}中{exam_class}考试的一道{question_type}，不需要做任何分析和解释，直接输出答案选项。\n{question}\n{option_str}'
        
        response = do_request(prompt)
        matches = re.findall("[ABCDE]", response)
        final_result = "".join(matches)
        print(final_result)
        
        info = {
                "id": test_id,
                "model_answer": final_result
            }
        results.append(info)
        c = c+1
        if c==58:
            time.sleep(65)
            c = 0
        
        time.sleep(5)
        
    with open('output.json', 'w', encoding="utf-8") as f1:
        json.dump(results, f1, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()


              