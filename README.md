# 全流程训练

## 1.安装依赖

```python
conda create -n llm python=3.11
conda activate llm
python -m pip install -r requirements.txt
```

## 2.数据配置

<details>
<summary>数据集配置、PT、SFT、RW数据格式详细内容</summary>

### dataset_info

如果您使用自定义数据集，请务必在 `dataset_info.json` 文件中以如下格式提供您的数据集定义。

```json
"数据集名称": {
  "hf_hub_url": "HuggingFace上的项目地址（若指定，则忽略下列三个参数）",
  "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略下列两个参数）",
  "file_name": "该目录下数据集文件的名称（若上述参数未指定，则此项必需）",
  "file_sha1": "数据集文件的SHA-1哈希值（可选）",
  "columns": {
    "prompt": "数据集代表提示词的表头名称（默认：instruction）",
    "query": "数据集代表请求的表头名称（默认：input）",
    "response": "数据集代表回答的表头名称（默认：output）",
    "history": "数据集代表历史对话的表头名称（默认：None）"
  }
}
```

其中 `prompt` 和 `response` 列应当是非空的字符串。`query` 列的内容将会和 `prompt` 列拼接作为模型输入。`history` 列应当是一个列表，其中每个元素是一个字符串二元组，分别代表用户请求和模型答复。

### PT example data

`.txt`格式，一行一个无监督数据。

```html
Machine learning (ML) is a field devoted to understanding and building methods that let machines "learn" – that is, methods that leverage data to improve computer performance on some set of tasks.
Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
```

### SFT example data 1

```json
[
  {
    "instruction": "听起来很不错。人工智能可能在哪些方面面临挑战呢？",
    "input": "",
    "output": "人工智能面临的挑战包括数据隐私、安全和道德方面的问题，以及影响就业机会的自动化等问题。",
    "history": [
      ["你好，你能帮我解答一个问题吗？", "当然，请问有什么问题？"],
      ["我想了解人工智能的未来发展方向，你有什么想法吗？", "人工智能在未来的发展方向可能包括更强大的机器学习算法，更先进的自然语言处理技术，以及更加智能的机器人。"]
    ]
  }
]
```

### SFT example data 2

```json
[
  {
    "instruction": "听起来很不错。人工智能可能在哪些方面面临挑战呢？",
    "input": "",
    "output": "人工智能面临的挑战包括数据隐私、安全和道德方面的问题，以及影响就业机会的自动化等问题。",
    "history": []
  }
]
```

### RW example data

```json
[
  {
    "instruction": "生成三个与“道歉”意思相同的动词",
    "input": "",
    "output": [
      "承认，表示遗憾，弥补。",
      "道歉"
    ]
  }
]
```
  
</details>

## 3.训练配置

<details>
<summary>训练参数与指令</summary>

### 配置分布式

```python
accelerate config # configure the environment
accelerate launch src/train_bash.py # arguments (same as above)
```

### 监督训练

```python
accelerate launch src/train_bash.py \
    --stage sft \
    --model_name_or_path ./Llama-2-7b-chat-hf \
    --do_train \
    --dataset mm \
    --finetuning_type lora \
    --quantization_bit 4 \
    --overwrite_cache \
    --output_dir output \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --fp16 \
    --template llama2 \
    --lora_target q_proj,v_proj
```
  
</details>

# 参考

- https://github.com/llSourcell/DoctorGPT
- https://github.com/facebookresearch/llama-recipes
- https://github.com/hiyouga/LLaMA-Efficient-Tuning
