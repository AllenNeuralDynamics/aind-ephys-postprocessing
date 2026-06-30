<<<<<<< HEAD
# hash:sha256:c9918fd2e205b11e9225c4ce45ce3ca1240d1ecfed937fb6779e6bbf689d238f
=======
# hash:sha256:36ebb09b5b3a3335f26b9ed8750ca5cdee00d1e57731c0ffd5b31e2850ac24b3
>>>>>>> dev2main
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
    aind-data-schema==2.8.1 \
    aind-log-utils==0.2.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.104.7 \
    wavpack-numcodecs==0.2.2
