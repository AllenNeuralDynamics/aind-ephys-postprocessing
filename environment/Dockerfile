<<<<<<< HEAD
# hash:sha256:726904eacdc83aafef6c46d36aba0cc802fd2110c4507f322fec35e8fd574886
=======
# hash:sha256:c9c40530acd06143dbfba4fb66c090a867d4e385f396a96e1a1d5b862343203a
>>>>>>> main
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
<<<<<<< HEAD
    aind-data-schema==0.38.5 \
    spikeinterface[full]==0.101.0rc1

COPY postInstall /
RUN /postInstall
=======
    aind-data-schema==0.38.0 \
    spikeinterface[full]==0.100.7
>>>>>>> main
