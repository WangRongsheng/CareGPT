## 全流程训练和推理

<details>
<summary>训练和推理参数与指令</summary>

### 配置分布式

```python
accelerate config # configure the environment
accelerate launch src/train_bash.py # arguments (same as above)
```

### 监督训练

```python
accelerate launch src/train_bash.py \
    --stage sft \
    --model_name_or_path ../Baichuan2-13B-Chat \
    --do_train \
    --dataset mm \
    --finetuning_type lora \
    --quantization_bit 4 \
    --overwrite_cache \
    --output_dir output-baichuan2-mm \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --fp16 \
    --template baichuan2 \
    --lora_target W_pack
```

### 推理

```python
python src/web_demo.py \
    --model_name_or_path ../Baichuan2-13B-Chat \
    --checkpoint_dir output-baichuan2-mm \
    --finetuning_type lora \
    --template baichuan2
```

</details>

## 开放权重

|阶段|权重介绍|下载地址|特点|底座模型|微调方法|数据集|
|:-|:-|:-|:-|:-|:-|:-|
|监督微调|多轮对话数据基于Baichuan2-13B-Chat训练而来|[⚙️Baichuan2-13B-Chat-sft-multi](https://huggingface.co/wangrongsheng/Baichuan2-13B-Chat-sft-multi)|出色的多轮对话能力|Baichuan2-13B-Chat|QLoRA|mm|

> 我们在[CMB](https://cmedbenchmark.llmzoo.com/)榜单进行了评测，评测脚本和评测结果见：[Eval](https://github.com/WangRongsheng/CareLlama/tree/carebaichuan2/eval)
