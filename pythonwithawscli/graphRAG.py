import boto3
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
import os
import litellm

litellm._turn_on_debug()


# AWS APIキーの設定
#os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'  # 使用するAWSリージョンを指定（例：us-east-1）

# boto3セッションの初期化（オプションでAWSのAPIキーを指定）
#session = boto3.Session(
#    region_name=os.getenv('AWS_DEFAULT_REGION')
#)
session = boto3.Session(profile_name='default')

print(os.getenv('AWS_DEFAULT_REGION'))
print(session.profile_name)
bedrock_client = session.client('bedrock-runtime')

# LLMとEmbeddingの設定
#llm = LiteLLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0")
#embedding = LiteLLMEmbedding(model_name="bedrock/amazon.titan-embed-text-v2:0")


llm = LiteLLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
              bedrock_client=bedrock_client)
embedding = LiteLLMEmbedding(model_name="bedrock/amazon.titan-embed-text-v2:0",
                             berrock_client=bedrock_client)

Settings.llm = llm
Settings.embed_model = embedding

# WikipediaReaderの使用
reader = WikipediaReader()
documents = reader.load_data(pages=["ラオウ"], lang_prefix="ja")

# Neo4jへの接続設定
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://neo4j:7687",
)

# PropertyGraphIndexの作成とデータ挿入
index = PropertyGraphIndex.from_existing(
    embed_model=embedding,
    kg_extractors=[
        SimpleLLMPathExtractor(),
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

for document in documents:
    index.insert(document)

