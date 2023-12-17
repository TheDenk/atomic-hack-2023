FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install \
    torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY . .
RUN pip3 install -U pip && pip install --no-cache-dir -r docker-requirements.txt

CMD [ "python3", "app.py"]