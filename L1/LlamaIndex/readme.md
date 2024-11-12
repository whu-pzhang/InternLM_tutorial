# 书生大模型实战 LlamaIndex RAG

所有实验基于 InternStudio 平台进行。

## 环境配置

```bash
# 创建环境
conda create -n llamaindex python=3.10 -y
conda activate llamaindex
# 安装依赖包
pip install llama-index==0.11.20
pip install llama-index-llms-replicate==0.3.0
pip install llama-index-llms-openai-like==0.2.0
pip install llama-index-embeddings-huggingface==0.3.1
pip install llama-index-embeddings-instructor==0.2.1
# InternStudio cuda12 环境运行报错，这里采用 cuda11.7 环境
# pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```


### 下载模型和数据

```bash
cd ~
mkdir llamaindex_demo model

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer

# 下载 NLTK 资源
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/* ./
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

## 基础任务


>- 任务要求1（必做，参考readme_api.md）：基于 LlamaIndex 构建自己的 RAG 知识库，寻找一个>问题 A 在使用 LlamaIndex 之前 浦语 API 不会回答，借助 LlamaIndex 后 浦语 API 具备回>答 A 的能力，截图保存。注意：写博客提交作业时切记不要泄漏自己 api_key！
>
>- 任务要求2（可选，参考readme.md）：基于 LlamaIndex 构建自己的 RAG 知识库，寻找一个问题 >A 在使用 LlamaIndex 之前 InternLM2-Chat-1.8B 模型不会回答，借助 LlamaIndex 后 >InternLM2-Chat-1.8B 模型具备回答 A 的能力，截图保存。
>
>- 任务要求3（优秀学员必做） ：将 Streamlit+LlamaIndex+浦语API的 Space 部署到 Hugging >Face。


### 基于 API 的 RAG

浦语官网和硅基流动都提供了InternLM的类OpenAI接口格式的免费的 API，可以访问以下两个了解两个 API 的使用方法和 Key。

- 浦语官方 API：https://internlm.intern-ai.org.cn/api/document
- 硅基流动：https://cloud.siliconflow.cn/models?mfs=internlm

首先我们测试不用 RAG ，直接询问 LLM ：

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# offical internlm
base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
api_key = os.getenv('PUYU_API_KEY')
model = 'internlm2.5-latest'

# # siliconflow
# base_url = "https://api.siliconflow.cn/v1"
# api_key = os.getenv('SILICON_API_KEY')
# model = 'internlm/internlm2_5-7b-chat'

client = OpenAI(api_key=api_key, base_url=base_url)

chat_rsp = client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'user', 'content': '硅基流动公司介绍'}
    ]
)

for choice in chat_rsp.choices:
    print(choice.message.content)
```

![](./llamaindex_01.jpg)


InternLM2.5 会直接回复不知道。


接下来，我们利用 LlamaxIndex 构建一个简单的 RAG 检索。

1. 找一段 硅基流动 公司的简介文字，存为 `/root/llamaindex_demo/data1/silicon_intro.md` 文件。
2. 填入一下代码，运行


```python
import os
os.environ['NLTK_DATA'] = '/root/nltk/data'

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.callbacks import CallbackManager
from llama_index.llms.openai_like import OpenAILike

from dotenv import load_dotenv

load_dotenv()

# create an instance of CallbackManager
callback_manager = CallbackManager()

api_base_url =  "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
model = "internlm2.5-latest"
api_key = os.environ['PUYU_API_KEY']

# # siliconflow
# base_url = "https://api.siliconflow.cn/v1"
# api_key = os.getenv('SILICON_API_KEY')
# model = 'internlm/internlm2_5-7b-chat'

llm = OpenAILike(
    model=model,
    api_base=api_base_url,
    api_key=api_key,
    is_chat_model=True,
    callback_manager=callback_manager
)

#
embed_model = HuggingFaceEmbedding(
    model_name='/root/model/sentence-transformer'
)
# 将创建的嵌入模型赋值给全局设置的 embed_model 属性，
# 这样在后续的索引构建过程中就会使用这个模型。
Settings.embed_model = embed_model

# 初始化 llm
Settings.llm = llm

# 从指定目录读取所有文档，并加载数据到内存中
documents = SimpleDirectoryReader('/root/llamaindex_demo/data1').load_data()
# 创建一个 VectorStoreIndex ，并使用之前加载的文档来构建索引
# 此索引将文档转换为向量，并存储这些向量以便于快速检索
index = VectorStoreIndex.from_documents(documents)
# 创建一个查询引擎，接收查询并返回相关文档的响应
query_engine = index.as_query_engine()
response = query_engine.query('给我介绍一下硅基流动（SiliconFlow）公司')

print(response)
```

![](./llamaindex_02.jpg)

可以看到，加入外挂知识库后，LLM 就可以比较准确的回答出之前不知道的问题了。

### 构建 Huggingface Space 应用


