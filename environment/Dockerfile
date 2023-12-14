# hash:sha256:0b00670a57cc539530bf1f9a766cba943edb4ab6147998b9a9286d39e249105e
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.22.1 \
    spikeinterface[full]==0.99.0

COPY postInstall /
RUN /postInstall
