import boto3
import os
import argparse
import logging
import tenacity
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.settings import Settings
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex, Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter # テキスト分割クラスをインポート

#logging.basicConfig(level=logging.DEBUG)

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

# PropertyGraphIndexの作成
index = PropertyGraphIndex.from_existing(
    embed_model=embedding,
    property_graph_store=graph_store,
    show_progress=True,
)


def load_documents(source_type, source_path):
    """入力ソースの種類に応じてドキュメントをロードする"""
    if source_type == "wiki":
        reader = WikipediaReader()
        documents = reader.load_data(pages=[source_path], lang_prefix="ja") # source_path がページタイトル
    elif source_type == "pdf":
        loader = PyPDFLoader(source_path) # source_path がファイル名
        loaded_documents = loader.load() # load() メソッドでロード
        # テキスト分割器 (CharacterTextSplitter) を作成 (ページごとに分割するための設定は要調整)
        text_splitter = CharacterTextSplitter(
            separator="\n\n", # ページ区切り文字 (PDF によって異なる場合あり)
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # ドキュメントをページ単位で分割 (実際には改行コードで分割されるため、ページ区切りが正確でない可能性あり)
        documents = text_splitter.split_documents(loaded_documents)

    elif source_type == "web":
        loader = WebBaseLoader(source_path) # source_path が URL
        documents = loader.load() # WebBaseLoader はデフォルトでドキュメント分割は行わない
    else:
        raise ValueError(f"Unknown source type: {source_type}")
    return documents


if __name__ == "__main__":
    # 引数パーサーの作成
    parser = argparse.ArgumentParser(description="知識グラフにデータを挿入して質問応答を実行します。",
                                     formatter_class=argparse.RawTextHelpFormatter) # ヘルプメッセージの改行を保持

    # 必須引数
    required_group = parser.add_argument_group('必須引数') # 引数グループ化
    required_group.add_argument("source_type", choices=["wiki", "pdf", "web"],
                                 help="入力ソースの種類 ('wiki', 'pdf', 'web' から選択)")
    required_group.add_argument("source_path",
                                 help="入力ソースのパス (wikiの場合はページタイトル, pdfの場合はファイル名, webの場合はURL)")
    required_group.add_argument("query", help="質問文字列")

    args = parser.parse_args()

    # ドキュメントのロード
    try:
        documents = load_documents(args.source_type, args.source_path)
    except ValueError as e:
        print(f"エラー: {e}")
        parser.print_help() # ヘルプメッセージを表示
        exit(1)
    except Exception as e: # ファイルが存在しない場合など
        print(f"ドキュメントのロード中にエラーが発生しました: {e}")
        parser.print_help()
        exit(1)


    # ドキュメントの挿入 (ページ単位で処理)
    for document in documents:
        # metadata に 'source' キーが存在しない場合に備えて、get メソッドとデフォルト値を使用
        source_info = document.metadata.get('source', '不明なソース')
        # ページ番号を取得 (metadata に page キーがあることを期待)
        page_num = document.metadata.get('page', '不明なページ番号')
        print(f"ドキュメント挿入中: {source_info}, ページ: {page_num}") # どのドキュメントのどのページを挿入しているか表示
        # テキストチャンクの分割 (例: 1000文字ごとに分割)
        chunk_size = 1000
        text_chunks = [document.page_content[i:i+chunk_size] for i in range(0, len(document.page_content), chunk_size)]
        for chunk in text_chunks:
            # 各チャンクを新しい Document オブジェクトとして挿入
            chunk_document = Document(page_content=chunk, metadata=document.metadata)
            index.insert(chunk_document)

    retriever = index.as_retriever(
        include_text=False,
    )

    results = retriever.retrieve(args.query)
    for record in results:
        print(record.text)

    query_engine = index.as_query_engine()

    response = query_engine.query(args.query)
    print(f"質問: {args.query}")
    print(f"回答: {response}")
