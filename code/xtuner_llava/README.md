# LLaVA 多模态模型微调



## 多模态LLM原理简介

### 文本单模态

```mermaid
flowchart LR
a[输入文本] -.- A(文本Embedding模型) -.-> b[/文本向量/]
b --> c((L L M))
c --> d[输出文本]
```

###  文本+图像多模态
```mermaid
flowchart LR
a[输入文本] -.- A(文本Embedding模型) -.-> b[/文本向量/]
b --> c((L L M))
c --> d[输出文本]
subgraph " "
f[输入图像] -.- F("Image Projector")
F -.-> g[/图像向量/]
end
g --> c
```

## LLaVA 

[Haotian Liu等](https://arxiv.org/abs/2304.08485)使用GPT-4V对图像数据生成描述，以此构建出大量`<question text><image> -- <answer text>`的数据对。利用这些数据对，配合文本单模态LLM，训练出一个Image Projector。

所使用的`文本单模型LLM`和训练出来的`Image Projector`，统称为`LLaVA模型`。

### LLaVA训练阶段示意图
```mermaid
flowchart TB;
subgraph 训练阶段
a[("文本+图像<br>问答对<br>(训练数据)")] --> b{显卡}
c((文本单模态<br>LLM)) --> b
b --> d([Image<br>Projector])
end
```

### LLaVA测试阶段示意图
```mermaid
flowchart TB;
subgraph 测试阶段
a([Image<br>Projector]) --> b{显卡}
c((文本单模态<br>LLM)) --> b
e[输入图像] --> b
b --> d[输出文本]
end
```

> Image Projector的训练和测试，有点类似之前我们讲过的LoRA微调方案。

二者都是在已有LLM的基础上，用新的数据训练一个新的小文件。

只不过，LLM套上LoRA之后，有了新的灵魂（角色）；而LLM套上Image Projector之后，才有了眼睛。

## 安装依赖

```
git clone https://github.com/InternLM/xtuner
cd xtuner
pip install -e '.[all]'
```

## 训练LLaVA模型

> 在本节中，我们将 **自己构造 `<question text><image>--<answer text>` 数据对，基于InternLM2_Chat_1.8B这个文本单模态模型，使用LLaVA方案，训练一个给InternLM2_Chat_1.8B使用的Image Projector文件。**

LLaVA方案中，给LLM增加视觉能力的过程，即是训练Image Projector文件的过程。
该过程分为2个阶段：Pretrain和Finetune。

```mermaid
flowchart LR;
    subgraph Pretrain阶段
    a[("图像<br>+<br>标题(短文本)")] --> b{显卡}
    c(("文本单模态LLM<br>(InternLM2_Chat_1.8B)")) --> b
    b --> d((Pretrained<br>LLaVA))
    end

    subgraph Finetune阶段
    f[("图像<br>+<br>复杂对话文本")] --> g{显卡}
    d --> g
    g --> i((Finetuned<br>LLaVA))
    end
```

### 1.3.3. Pretrain阶段
在Pretrain阶段，我们会使用大量的`图片+简单文本（caption, 即图片标题）`数据对，使LLM理解图像中的**普遍特征**。即，对大量的图片进行**粗看**。

Pretrain阶段训练完成后，此时的模型已经有视觉能力了！但是由于训练数据中都是图片+图片标题，所以此时的模型虽然有视觉能力，但无论用户问它什么，它都只会回答输入图片的标题。即，**此时的模型只会给输入图像“写标题”**。

> Pretrain阶段相当于是开发LLM时预训练工作，对硬件要求非常高，有8卡的学有余力同学可以自行尝试。详见[XTuner-LLaVA](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/user_guides/dataset_prepare.md#llava-dataset)和[LLaVA](https://llava-vl.github.io/)。
> <details>
> 
> ```bash
> NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
> 
> NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune --deepspeed deepspeed_zero2
> ```

> </details>

在本次实战营中，我们已经为大家提供了Pretrain阶段的产物——`iter_2181.pth`文件。它就是幼稚园阶段的Image Projector！大家带着`iter_2181.pth`文件继续进入下一阶段进行Finetune即可。

### 1.3.4. Finetune阶段
在Finetune阶段，我们会使用`图片+复杂文本`数据对，来对Pretrain得到的Image Projector即iter_2181.pth进行进一步的训练。

#### 1.3.4.1. 训练数据构建

##### 1.3.4.1.1. 格式
```json
[
    {
        "id": "随便什么字符串",
        "image": "图片文件的相对位置。相对谁？相对你后面config文件里指定的image_folder参数的路径。",
        "conversation": [
            {
                "from": "human",
                "value": "<image>\n第1个问题。"
            },
            {
                "from": "gpt",
                "value": "第1个回答"
            },
            {
                "from": "human",
                "value": "第2个问题。"
            },
            {
                "from": "gpt",
                "value": "第2个回答"
            },
            # ......
            {
                "from": "human",
                "value": "第n个问题。"
            },
            {
                "from": "gpt",
                "value": "第n个回答"
            },
        ]
    },

    # 下面是第2组训练数据了。

    {
        "id": "随便什么字符串",
        "image": "图片文件的相对位置。相对谁？相对你后面config文件里指定的image_folder参数的路径。",
        "conversation": [
            {
                "from": "human",
                "value": "<image>\n第1个问题。"
            },
            # ......
            {
                "from": "gpt",
                "value": "第n个回答"
            }
        ]
    }
]
```
> 注意：每组训练数据的第1个来自human的问题前，要加上图片占位符，即`<image>`

<details>
<summary>示例</summary>

```json
[
  {
    "id": "000000033471",
    "image": "coco/train2017/000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  },
  {
    "id": "000000052846",
    "image": "coco/train2017/000000052846.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhere is the cat positioned in the image?"
      },
      {
        "from": "gpt",
        "value": "The cat is positioned on top of the back of the couch in the living room."
      }
    ]
  }
 ]
```
</details>

##### 1.3.4.1.2. 制作
我们可以效法LLaVA作者的做法，将自己的图片发送给GPT，要求其按照上述格式生成若干条问答对。
<details>
<summary>prompts</summary>

![001.jpg](llava_data/images/00008.jpg)

Create a dataset for me, following this format.
```json
[
  {
    "id": "<random_number_string>",
    "image": "images/00008.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe this image."
      },
      {
        "from": "gpt",
        "value": "<answer1>"
      },
      {
        "from": "human",
        "value": "<question2>"
      },
      {
        "from": "gpt",
        "value": "<answer2>"
      },
      {
        "from": "human",
        "value": "<question3>"
      },
      {
        "from": "gpt",
        "value": "<answer3>"
      },
      {
        "from": "human",
        "value": "<question4>"
      },
      {
        "from": "gpt",
        "value": "<answer4>"
      }
    ]
  }
]
```
The questions and answers, please generate for me, based on the image I sent to you. Thes questions should be from the shallow to the deep, and the answers should be as detailed and correct as possible. The questions and answers should be stick to the contents in the image itself, like objects, peoples, equipment, environment, purpose, color, attitude, etc. 5 question and answer pairs.
</details>
<br>


