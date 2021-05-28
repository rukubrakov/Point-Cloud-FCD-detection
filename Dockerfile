FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt update && apt install python3-pip -y

COPY ./requirements.txt /code/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r /code/requirements.txt && \
    sync 
