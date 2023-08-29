import os
from functools import partial
import gradio as gr
import openai

class Messages_lst:
    def __init__(self):
        self.memory = []
    
    def update(self, role,message):
        if role == "user":
            user_turn = {"role": "user","content":message}
            self.memory.append(user_turn)
        elif role == "assistant":
            gpt_turn = {"role": "assistant","content":message}
            self.memory.append(gpt_turn)
    
    def get(self):
        return self.memory
    
messages_lst = Messages_lst()

def get_response(api_key_input, user_input):
    
    # print(api_key_input)
    print(user_input)

    messages_lst.update("user", user_input)
    messages = messages_lst.get()

    openai.api_key = api_key_input
    MODEL = "gpt-3.5-turbo"

    print(messages)

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages = messages,
        temperature=0.5)
    assistant = response['choices'][0]['message']['content']
    messages_lst.update("assistant", assistant)
    # return assistant
    # 生成HTML字符串
    html_string = ""
    for message in messages_lst.get():
        if message["role"] == "user":
            html_string += f"<p><b>User:</b> {message['content']}</p>"
        else:
            html_string += f"<p><b>Assistant:</b> {message['content']}</p>"

    return html_string

def main():
    # api_key = os.environ.get("OPENAI_API_KEY")

    api_key_input = gr.components.Textbox(
        lines=1,
        label="Enter OpenAI API Key",
        type="password",
    )

    user_input = gr.components.Textbox(
        lines=3,
        label="Enter your message",
    )
    

    output_history = gr.outputs.HTML(
        label="Updated Conversation",
    )

    inputs = [
        api_key_input,
        user_input,
    ]

    iface = gr.Interface(
        fn=get_response,
        inputs=inputs,
        outputs=[output_history],
        title="GPT WebUi",
        description="A simple chatbot using Gradio",
        allow_flagging="never",
    )

    iface.launch()


if __name__ == '__main__':
    main()