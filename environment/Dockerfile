# hash:sha256:51b80d27b9fb74195faa7cbb94f671c32f90dcca76db8f9090f3435d4b470fb1
FROM registry.codeocean.allenneuraldynamics-test.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.13.66 \
    numba==0.57.0

COPY postInstall /
RUN /postInstall
