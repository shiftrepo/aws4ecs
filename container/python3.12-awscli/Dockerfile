# Python 3.12 ベースイメージ
FROM python:3.12-slim

# 環境変数の設定
ENV AWS_CLI_VERSION=2.15.39 \
    PATH="/usr/local/aws-cli/v2/current/bin/:$PATH"

# 必要なツールをインストールし、AWS CLI をセットアップ
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    groff \
    less \
    git \
    vim \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && aws --version \
    && rm -rf awscliv2.zip aws \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 動作確認用デフォルトコマンド
CMD ["aws", "--version"]
