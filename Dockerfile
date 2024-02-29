FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .