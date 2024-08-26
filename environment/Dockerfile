# hash:sha256:a2e7bfb4d06fbda174864ce4a0b2632cc057b160422367a711e15a077098bd1f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.5 \
    spikeinterface[full]==0.101.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
