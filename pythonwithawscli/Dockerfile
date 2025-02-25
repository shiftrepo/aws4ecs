FROM public.ecr.aws/amazonlinux/amazonlinux:2 as installer
ARG EXE_FILENAME=awscli-exe-linux-x86_64.zip

RUN yum update -y \
  && yum install -y unzip curl \
  && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# aws cli
COPY $EXE_FILENAME .
RUN yum update -y \
  && yum install -y unzip \
  && unzip $EXE_FILENAME \
  # The --bin-dir is specified so that we can copy the
  # entire bin directory from the installer stage into
  # into /usr/local/bin of the final stage without
  # accidentally copying over any other executables that
  # may be present in /usr/local/bin of the installer stage.
  && ./aws/install --bin-dir /aws-cli-bin/

# software
FROM public.ecr.aws/amazonlinux/amazonlinux:2
RUN yum update -y \
  && yum install -y less groff tar unzip vi\
  && yum clean all
COPY --from=installer /usr/local/aws-cli/ /usr/local/aws-cli/
COPY --from=installer /aws-cli-bin/ /usr/local/bin/

# python
COPY requirements.txt /requirements.txt
RUN yum install -y python3 python-pip && yum clean all
RUN python3 -m pip install --upgrade pip --no-cache-dir && python3 -m pip --no-cache install -r /requirements.txt --no-cache-dir

WORKDIR /aws
ENTRYPOINT ["/usr/local/bin/aws"]
