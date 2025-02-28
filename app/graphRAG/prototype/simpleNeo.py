import boto3
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

import os

# boto3セッションの初期化（オプションでAWSのAPIキーを指定）
session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
 aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
 region_name=os.getenv('AWS_DEFAULT_REGION')
)
#session = boto3.Session(profile_name='default')

print(os.getenv('AWS_DEFAULT_REGION'))
#print(session.profile_name)
bedrock_client = session.client('bedrock-runtime')

# LLMとEmbeddingの設定
llm = Bedrock(model="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client)
embedding = BedrockEmbedding(model_name="amazon.titan-embed-text-v2:0", client=bedrock_client)

Settings.llm = llm
Settings.embed_model = embedding

# WikipediaReaderの使用
reader = WikipediaReader()
documents = reader.load_data(pages=["ラオウ"], lang_prefix="ja")

# Neo4jへの接続設定 (PropertyGraphIndex.from_documents 内で自動作成されるため、明示的な graph_store の初期化は不要)
# graph_store = Neo4jPropertyGraphStore(
#     username="neo4j",
#     password="password",
#     url="bolt://neo4j:7687",
# )

# PropertyGraphIndexの作成とデータ挿入 (from_documents を使用)
index = PropertyGraphIndex.from_documents(
        documents=documents,
        embed_model=embedding,
   show_progress=True,
)

# for document in documents: # from_documents の場合、insert は不要
#     index.insert(document)
retriever = index.as_retriever(
    include_text=False,
)

query = "ラオウの家族の名前を教えて"

results = retriever.retrieve(query)
for record in results:
    print(record.text)

query_engine = index.as_query_engine()

response = query_engine.query(query)
print(response)


