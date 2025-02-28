FROM public.ecr.aws/amazonlinux/amazonlinux:2 as installer
ARG EXE_FILENAME=awscli-exe-linux-x86_64.zip

# AWS CLI インストール準備
RUN yum update -y \
    && yum install -y unzip curl \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install --bin-dir /aws-cli-bin/ \
    && yum clean all

# 最終イメージ作成
FROM public.ecr.aws/amazonlinux/amazonlinux:2

# 必要ツールのインストール
RUN yum update -y \
    && yum install -y less groff tar vi wget gcc make \
    zlib-devel bzip2 bzip2-devel readline-devel sqlite-devel \
    openssl-devel xz xz-devel libffi-devel \
    && yum groupinstall -y "Development Tools" \
    && yum clean all

# AWS CLI のコピー
COPY --from=installer /usr/local/aws-cli/ /usr/local/aws-cli/
COPY --from=installer /aws-cli-bin/ /usr/local/bin/

# Python 3.12 インストール
RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz \
    && tar xzf Python-3.12.2.tgz \
    && cd Python-3.12.2 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && rm -rf /usr/src/Python-3.12.2*  # 後処理でクリーンアップ

# Python3.12 を python3 としてリンク
RUN ln -s /usr/local/bin/python3.12 /usr/bin/python3 \
    && ln -s /usr/local/bin/pip3.12 /usr/bin/pip3

# Python 確認用
RUN python3 --version && pip3 --version

# requirements インストール
COPY requirements.txt /requirements.txt
#RUN pip3 install --upgrade pip --no-cache-dir \
#    && pip3 install -r /requirements.txt --no-cache-dir

WORKDIR /aws
CMD ["/bin/bash"]
