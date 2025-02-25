FROM public.ecr.aws/amazonlinux/amazonlinux:2 as installer
ARG EXE_FILENAME=awscli-exe-linux-x86_64.zip

# AWS CLIインストール
RUN yum update -y \
  && yum install -y unzip curl \
  && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
  && unzip awscliv2.zip \
  && ./aws/install --bin-dir /aws-cli-bin/

# Python & その他インストール
FROM public.ecr.aws/amazonlinux/amazonlinux:2

# 必要パッケージインストール
RUN yum update -y \
  && yum install -y less groff tar vi openssl openssl-devel libffi-devel bzip2-devel \
  zlib-devel readline-devel sqlite-devel xz-devel gcc make wget gzip \
  && yum clean all

COPY --from=installer /usr/local/aws-cli/ /usr/local/aws-cli/
COPY --from=installer /aws-cli-bin/ /usr/local/bin/

COPY requirements.txt /requirements.txt

# Pythonのインストール
RUN cd /usr/src \
  && wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz \
  && tar xzf Python-3.12.2.tgz \
  && cd Python-3.12.2 \
  && ./configure --enable-optimizations --with-openssl=/usr \
  && make altinstall \
  && ln -s /usr/local/bin/python3.12 /usr/bin/python3 \
  && ln -s /usr/local/bin/pip3.12 /usr/bin/pip3

# pipインストールとパッケージ取得
RUN python3 -m ensurepip --upgrade \
  && pip3 install --upgrade pip --no-cache-dir \
  && pip3 install -r /requirements.txt --no-cache-dir

WORKDIR /aws
CMD ["/bin/bash"]

