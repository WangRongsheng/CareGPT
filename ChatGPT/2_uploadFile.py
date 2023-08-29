import requests
import openai

url = "https://api.openai.com/v1/files"
headers = {
    "Authorization": "Bearer $OPENAI_API_KEY"
}

payload = {
    "purpose": "fine-tune",
}
files = {
    "file": open("/Users/lhj/AI/openai_cookbook/output.jsonl", "rb")
}

response = requests.post(url, headers=headers, data=payload, files=files)
print(response)

openai.api_key = $OPENAI_API_KEY
print(openai.File.list())
