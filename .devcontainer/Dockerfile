FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install notebook
RUN pip install scikit-learn pandas matplotlib seaborn 
RUN pip install transformers evaluate
RUN pip install tqdm
RUN pip install requests
RUN pip install pydantic
RUN pip install python-dotenv