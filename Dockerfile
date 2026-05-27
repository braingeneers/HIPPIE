#Pytorch dockerfile
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

USER root

WORKDIR /src

RUN apt-get update && apt-get install -y nano && rm -rf /var/lib/apt/lists/*

# Install the hippie package and its declared dependencies (from pyproject.toml)
COPY pyproject.toml README.md ./
COPY hippie ./hippie
RUN pip install --no-cache-dir -e .

# Copy the user-facing entry points and bundled example data
COPY hippie_nwb_classify.py ./
COPY examples ./examples
COPY datasets_hippie ./datasets_hippie