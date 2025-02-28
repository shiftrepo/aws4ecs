import boto3
import os
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.settings import Settings
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
import logging

#logging.basicConfig(level=logging.DEBUG)

#from llama_index.core import set_global_handler

#def debug_handler(level, message):
#    print(f"{level}: {message}")

#set_global_handler(debug_handler)

# boto3セッションの初期化（同期版を使用）
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

bedrock_client = session.client('bedrock-runtime')

#  で  を指定する
llm = Bedrock(model="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client)
embedding = BedrockEmbedding(model_name="amazon.titan-embed-text-v2:0", client=bedrock_client, use_async=False)

Settings.llm = llm
Settings.embed_model = embedding

# WikipediaReaderの使用
reader = WikipediaReader()
documents = reader.load_data(pages=["ルーク・スカイウォーカー"], lang_prefix="ja")

# Neo4jへの接続設定
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://neo4j:7687",
)

# PropertyGraphIndexの作成
index = PropertyGraphIndex.from_existing(
    embed_model=embedding,
    property_graph_store=graph_store,
    show_progress=True,
)

#query_str = index.as_query_str()
#print(query_str)



# ドキュメントの挿入
for document in documents:
    print(document)
    index.insert(document)


retriever = index.as_retriever(
    include_text=False,
)


query = "ルーク・スカイウォーカーの家族の名前を教えて"

results = retriever.retrieve(query)
for record in results:
    print(record.text)

query_engine = index.as_query_engine()

response = query_engine.query(query)
print(response)
