# docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    libsox-fmt-all \
    sox \
    libsox-dev

# Install python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Exponer puerto para jupyter
EXPOSE 8888

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run interactive shell
CMD ["/bin/bash"]

