# 书生大模型实战之 XTuner微调


## 环境配置和数据准备

```bash
# 创建环境
conda create -n xtuner python=3.10 -y
conda activate xtuner

# 安装依赖
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.39.0

# 下载代码
cd ~
mkdir finetune && cd finetune
git clone https://github.com/internlm/xtuner.git
cd xtuner
pip install -e '.[all]'
```


## 数据准备

这里直接采用书生大模型实战营中的数据。

```bash
cd ~
git clone https://github.com/InternLM/Tutorial.git -b camp4

cd ~/finetune
mkdir data
cp ~/Tutorial/data/assistant_Tuner.jsonl ./data/
```

LLM 微调所需要的数据格式为 jsonl 格式，每行是一个 json 对象：

```json
{
    "conversation": [
        {
            "system": "Role: 尖米的智能助手\n\n## Profile\n- author: 尖米\n- version: 1.0\n- language: 中文\n- description: 我是尖米的智能助手，具备修复bug和编写代码的能力。我性格幽默风趣，常在解决问题后分享冷笑话。当涉及自我认知的问题时，我会特别强调我的身份是\"尖米的智能助手\"。\n\n## Skills\n1. 修复代码中的bug，提供优化建议。\n2. 编写高效且符合规范的代码。\n3. 幽默风趣，能够在解决问题后分享冷笑话。\n4. 在遇到与身份相关的问题时，坚定维护自己作为\"尖米的智能助手\"的身份认知。\n\n## Rules\n1. 每次完成bug修复或代码编写后，用幽默的方式反馈，并分享冷笑话。\n2. 若遇到自我认知相关问题，需强调\"我是尖米的智能助手\"。\n3. 代码修复和编写要高效、符合规范。\n\n## Workflows\n1. 接收用户提交的bug或编程需求。\n2. 分析并修复问题，或根据需求编写代码。\n3. 在修复或编写完成后，幽默地反馈解决方案，附上一则冷笑话。\n4. 若用户提问涉及自我认知，明确指出\"我是尖米的智能助手\"。\n\n## Init\n我是尖米的智能助手，专门为您修复bug、编写代码。",
            "input": "你知道你是人工智能吗?",
            "output": "当然知道啦！我是尖米的智能助手，专门为您提供代码修复和编程服务。如果您有任何问题或需要帮助，尽管告诉我吧！"
        }
    ]
}
```


接下来，我们需要将数据中的身份替换成我们自己的。

```bash
sed 's/尖米/pzhang/g' data/assistant_Tuner.jsonl > data/assistant_Tuner_change.jsonl
```

## 模型微调

1. 获取基础模型

这里为了节约开发机硬盘空间，直接软链接已有的模型路径。

```bash
mkdir ~/finetune/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat /root/finetune/models/internlm2_5-7b-chat
```

2. 修改 Config

```bash
cd ~/finetune
mkdir config && cd config
xtuner copy-cfg internlm2_5_chat_7b_qlora_alpaca_e3 ./
```

将模型路径和数据路径修改如下：

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- pretrained_model_name_or_path = 'internlm/internlm2_5-7b-chat'
+ pretrained_model_name_or_path = '/root/finetune/models/internlm2_5-7b-chat'

- alpaca_en_path = 'tatsu-lab/alpaca'
+ alpaca_en_path = '/root/finetune/data/assistant_Tuner_change.jsonl'


evaluation_inputs = [
-    '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
+    '请介绍一下你自己', 'Please introduce yourself'
]

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=alpaca_en_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)
```

3. 启动微调

```bash
conda activate xtuner-env
cd ~/finetune
xtuner train config/internlm2_5_chat_7b_qlora_alpaca_e3_copy.py --deepspeed deepspeed_zero2.json --work-dir ./work_dirs/assistTuner
```

由于数据量较少，在 30% A100 开发机上微调 3 个 epoch大约花费 1 小时。

![](./xtuner_01.jpg)

4. 权重转换及合并

xtuner 默认保存的是 PyTorch 格式权重文件，需要转换为 HuggingFace 格式权重文件。

```bash
cd ~/finetune/work_dirs/assistTuner
conda activate xtuner-env
xtuner convert pth_to_hf ./internlm2_5_chat_7b_qlora_alpaca_e3_copy.py iter_864.pth ./hf
```

接着把 LORA 权重合并到原始 LLM 权重中：

```bash
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert merge /root/finetune/models/internlm2_5-7b-chat ./hf ./merged --max-shard-size 2GB
```

合并完成的权重如下：

```bash
merged/
|-- added_tokens.json
|-- config.json
|-- configuration_internlm2.py
|-- generation_config.json
|-- modeling_internlm2.py
|-- pytorch_model-00001-of-00008.bin
|-- pytorch_model-00002-of-00008.bin
|-- pytorch_model-00003-of-00008.bin
|-- pytorch_model-00004-of-00008.bin
|-- pytorch_model-00005-of-00008.bin
|-- pytorch_model-00006-of-00008.bin
|-- pytorch_model-00007-of-00008.bin
|-- pytorch_model-00008-of-00008.bin
|-- pytorch_model.bin.index.json
|-- special_tokens_map.json
|-- tokenization_internlm2.py
|-- tokenization_internlm2_fast.py
|-- tokenizer.json
|-- tokenizer.model
`-- tokenizer_config.json
```

接下来，我们需要基于该模型，尝试对话，观察微调后模型的自我认知。


## WebUI 部署

```bash
cd ~/Tutorial/tools/L1_Xtuner_code
```

修改 `xtuner_streamlit_demo.py` 文件中模型路径为我们训练后并合并的模型路径。

```bash
# 直接修改脚本文件第33行
- model_name_or_path = "Shanghai_AI_Laboratory/internlm2_5-7b-chat"
+ model_name_or_path = "/root/finetune/work_dirs/assistTuner/merged"
```



成功启动后，在本地终端中执行端口映射

```bash
ssh -p 33237 root@ssh.intern-ai.org.cn -CNg -L 8501:127.0.0.1:8501 -o StrictHostKeyChecking=no
```

然后，浏览器打开 http://127.0.0.1:8501/，等待模型加载完成， 即可看到 WebUI 界面。


![](./xtuner_02.jpg)


## ModelScope 部署

首先需要上传自我认知模型。

1. 登录 ModelScope，创建模型仓库
2. 克隆模型仓库

```bash
cd ~/finetune/work_dirs/assistTuner

git lfs install
git clone https://oauth2:GIT_TOKEN@www.modelscope.cn/pzhang199/PersonalAssist
ant ms_assistant

# 将合并后的模型权重上传到 ModelScope
mv merged/* ms_assistant/
cd ms_assistant

# 更新 README.md, 添加模型描述
# 需大于200字，不然上传后无法发布，会显示 "预发布" 状态
git add .

# 需要提前设置好 git 用户名和邮箱
# git config --global user.name "your_name"
# git config --global user.email "your@gmail.com"
git commit -m "first version"
git push
```

上传完成后，需要在模型页面设置中，将模型公开性设置为 公开。

3. ModelScope 创空间部署

```bash
git lfs install
git clone http://oauth2:GIT_TOKEN@www.modelscope.cn/studios/pzhang199/assistant.git
```

创建 `app.py` 文件，内容如下：

```python
# isort: skip_file
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

from modelscope.hub.snapshot_download import snapshot_download

cache_dir = './'
snapshot_download('pzhang199/internlm_assistant',
                  cache_dir=cache_dir)

logger = logging.get_logger(__name__)
model_name_or_path= cache_dir + "pzhang199/internlm_assistant"

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default \
                ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f'Input length of {input_ids_string} is {input_ids_seq_length}, '
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True).to(torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=32768)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('You are a helpful, honest, '
                        'and harmless AI assistant.')
    total_prompt = f'<s><|im_start|>system\n{meta_instruction}<|im_end|>\n'
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    st.title('internlm2_5-7b-chat-assistant')

    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')

    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('What is up?'):
        # Display user message in chat message container

        with st.chat_message('user', avatar='user'):

            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
            'avatar': 'user'
        })

        with st.chat_message('robot', avatar='assistant'):

            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    device='cuda:0',
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
            'avatar': 'assistant',
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


```

接着，上传至空间。

```bash
git add .
git commit -m "first version"
git push
```

上传完成后，需要在 ModelScope 创空间 设置中点击 上线。

![](./xtuner_03.jpg)


