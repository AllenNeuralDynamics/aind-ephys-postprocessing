# hash:sha256:9be7f802428402a6d0369296779013e4d53fca2f87e37087dc1ec2052994f3c7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.103.2 \
    wavpack-numcodecs==0.2.2
