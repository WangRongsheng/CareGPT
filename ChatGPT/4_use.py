import requests

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer $OPENAI_API_KEY"
}
data = {
    "model": "ft:gpt-3.5-turbo-0613:XXXXXXX",
    "messages": [
        {
            "role": "system",
            "content": "You are an assistant that occasionally misspells words"
        },
        {
            "role": "user",
            "content": "我在体检是正常的，但是去献血医生最是说我的血压高，不能献。血压是130、80这是为什么呢？"
        }
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response['choices'][0]['text'])
