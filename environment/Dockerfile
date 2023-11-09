# hash:sha256:f5487aca5571fead0be8271cacafdc74afad12ba0dd80561aa280e0f92a1b580
FROM registry.codeocean.allenneuraldynamics.org/codeocean/py-r:python3.9-R4.2.2-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.15.25

COPY postInstall /
RUN /postInstall
