import boto3
import os
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.settings import Settings
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
import logging
import tenacity  # リトライ処理用ライブラリ

logging.basicConfig(level=logging.DEBUG)

# boto3セッションの初期化
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

bedrock_client = session.client('bedrock-runtime')

# LLMとEmbeddingの設定
llm = Bedrock(model="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client)

# タイムアウト時間を延長 (秒単位で指定、例: 60秒)
request_timeout_sec = 60

# リトライ処理を Tenacity で実装
@tenacity.retry(stop=tenacity.stop_after_attempt(3),  # 最大3回リトライ
                  wait=tenacity.wait_fixed(5),       # リトライ間隔を5秒に固定
                  retry=tenacity.retry_if_exception_type(Exception), # すべての例外でリトライ
                  before_sleep=tenacity.before_sleep_log(logging, logging.WARNING)) # リトライ前にログ出力
def create_embedding_with_retry(client, model_name, request_timeout):
    return BedrockEmbedding(
        model_name=model_name,
        client=client,
        use_async=False,
        request_timeout=request_timeout # タイムアウト時間を設定
    )

embedding = create_embedding_with_retry(
    bedrock_client,
    "amazon.titan-embed-text-v2:0",
    request_timeout_sec
)


Settings.llm = llm
Settings.embed_model = embedding

# WikipediaReaderの使用
reader = WikipediaReader()
documents = reader.load_data(pages=["ダース・ベイダー"], lang_prefix="ja")

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


# ドキュメントの挿入
for document in documents:
    print(document)
    # テキストチャンクの分割 (例: 1000文字ごとに分割)
    chunk_size = 1000
    text_chunks = [document.text[i:i+chunk_size] for i in range(0, len(document.text), chunk_size)]
    for chunk in text_chunks:
        # 各チャンクを新しい Document オブジェクトとして挿入
        chunk_document = type(document)(text=chunk, metadata=document.metadata) # ドキュメントクラスを維持
        index.insert(chunk_document)


retriever = index.as_retriever(
    include_text=False,
)

query = "ダース・ベイダーの家族の名前を教えて"

results = retriever.retrieve(query)
for record in results:
    print(record.text)

query_engine = index.as_query_engine()

response = query_engine.query(query)
print(response)
