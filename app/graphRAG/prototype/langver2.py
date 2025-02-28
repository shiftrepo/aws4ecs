import os
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import boto3

# boto3セッションの初期化
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

bedrock_client = session.client('bedrock-runtime')

# LLMとEmbeddingの設定
llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client)
embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

# WikipediaLoaderの使用
loader = WikipediaLoader(query="ラオウ", lang="ja", load_max_docs=1)
documents = loader.load()

# Neo4jへの接続設定
url = "bolt://neo4j:7687"
username = "neo4j"
password = "password"
graph = Neo4jGraph(url=url, username=username, password=password)

#  知識グラフの構築 (Langchainでは知識グラフの自動構築は直接的にはサポートされていません。
#  以下は、シンプルな例として、ドキュメントからエンティティと関係を抽出するプロンプトの例です。
#  より複雑な知識グラフの構築には、追加のNLP処理や専用の知識グラフ構築パイプラインが必要になる場合があります。)
#  ここでは例として簡易的なプロンプトとChainを使用して、ノードとリレーションをNeo4jに登録する例を示します。
#  **このコードは知識グラフを自動構築するものではなく、例示です。**

#  より高度な知識グラフ構築を行う場合は、
#  1.  NER (固有表現抽出) モデルでエンティティを抽出
#  2.  関係抽出モデルでエンティティ間の関係を抽出
#  3.  抽出されたエンティティと関係をNeo4jに登録
#  のようなパイプラインを構築する必要があります。

def build_graph_from_documents(documents, graph, llm):
    for document in documents:
        content = document.page_content
        prompt = PromptTemplate.from_template("""
            与えられたテキストからエンティティと関係を抽出し、Neo4jのCypherクエリに変換してください。
            エンティティはノードとして、関係はリレーションシップとして表現してください。
            テキスト: {text}
            """)
        cypher_generation_chain = prompt | llm
        cypher_query = cypher_generation_chain.invoke({"text": content})

        #  **重要**: 生成されたCypherクエリは例示であり、実際のテキスト内容に合わせて調整が必要です。
        #  また、この例では簡易的なプロンプトを使用しているため、複雑なテキストに対しては適切なクエリが生成されない可能性があります。
        #  より高度な知識グラフ構築には、NERと関係抽出モデルの導入と、それらをLangchainで連携させる必要があります。

        try:
            graph.query(cypher_query.content) #  Bedrock LLMのResponseContentは.contentでアクセス
            print("Cypherクエリ実行成功")
            print(cypher_query.content)

        except Exception as e:
            print(f"Cypherクエリ実行失敗: {e}")
            print(cypher_query.content)


# ドキュメントからグラフを構築 (簡易的な例)
build_graph_from_documents(documents, graph, llm)


#  **注意点**
#  * 上記のコードは、LangchainでNeo4jに接続し、Wikipediaからロードしたドキュメントを基に、**簡易的な知識グラフ**を構築する例です。
#  * 知識グラフの自動構築はLangchainの主要な機能ではありません。より高度な知識グラフを構築するには、NERや関係抽出などのNLP技術とLangchainを組み合わせる必要があります。
#  * 上記コードの  関数内のプロンプトとCypherクエリ生成部分は**例示**です。実際のテキストと構築したい知識グラフに合わせて、プロンプトを調整し、より高度なエンティティ・関係抽出パイプラインを構築する必要があります。
#  *   と  は  に直接的な代替機能はありません。 で同様の機能を実現するには、より複雑なChainやAgentの構築が必要になる場合があります。

#  **より高度な知識グラフ活用例 (GraphCypherQAChain)**
#  以下は、構築したグラフに対して質問応答を行う例です。

graph_cypher_qa_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

#  Neo4j にデータを投入後、質問応答を実行
query = "ラオウは誰ですか？" #  日本語で質問
# DeprecationWarning: Chain.run is deprecated, use .invoke() instead
result = graph_cypher_qa_chain.invoke({"query": query})
print(f"質問: {query}")
print(f"回答: {result}")
