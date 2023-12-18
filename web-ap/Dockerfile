FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN chmod 1777 /tmp
RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 libsndfile1 -y

COPY . .
RUN pip3 install -U pip && pip install --no-cache-dir -r docker-requirements.txt
RUN pip3 install \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

CMD [ "python3", "app.py"]
