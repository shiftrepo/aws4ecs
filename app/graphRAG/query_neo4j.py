import boto3
import os
import argparse
import logging
import tenacity
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.settings import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex

# ログレベルを INFO に設定 (必要に応じて変更可能)
logging.basicConfig(level=logging.INFO)

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
@tenacity.retry(stop=tenacity.stop_after_attempt(3),
                  wait=tenacity.wait_fixed(5),
                  retry=tenacity.retry_if_exception_type(Exception),
                  before_sleep=tenacity.before_sleep_log(logging, logging.WARNING))
def create_embedding_with_retry(client, model_name, request_timeout):
    return BedrockEmbedding(
        model_name=model_name,
        client=client,
        use_async=False,
        request_timeout=request_timeout
    )

embedding = create_embedding_with_retry(
    bedrock_client,
    "amazon.titan-embed-text-v2:0",
    request_timeout_sec
)

Settings.llm = llm
Settings.embed_model = embedding

# Neo4jへの接続設定
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="password",
    url="bolt://neo4j:7687",
)

# PropertyGraphIndexの作成 (既存のインデックスからロード)
index = PropertyGraphIndex.from_existing(
    embed_model=embedding,
    property_graph_store=graph_store,
    show_progress=True,
)


if __name__ == "__main__":
    # 引数パーサーの作成 (質問文字列のみ)
    parser = argparse.ArgumentParser(description="知識グラフに質問応答を実行します。",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # 必須引数 (質問文字列)
    required_group = parser.add_argument_group('必須引数')
    required_group.add_argument("query", help="質問文字列")

    args = parser.parse_args()

    retriever = index.as_retriever(
        include_text=False,
    )

    results = retriever.retrieve(args.query)
    for record in results:
        print(record.text)

    # クエリエンジンの作成
    query_engine = index.as_query_engine()

    # 質問応答を実行
    response = query_engine.query(args.query)

    # 質問と回答を標準出力に表示
    print(f"質問: {args.query}")
    print(f"回答: {response}")
