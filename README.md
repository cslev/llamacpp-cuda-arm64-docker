# llamacpp-cuda-arm64-docker
This repo is to build your own llama.cpp Dockerimage with CUDA for ARM64 (DGX Spark). 


In order to compile this Docker image, you have to have a fully working NVidia ARM64-based Linux system with all drivers, including CUDA too. It also means, you are actually able to 
run llama.cpp or any AI workload efficiently on your system by offloading (most of) the processing to the GPU entirely.

## Best way to check
The best way to check is to run certain commands and verify their outputs:
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Nov__7_07:24:07_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.80
Build cuda_13.1.r13.1/compiler.36836380_0

$ nvidia-smi 
Wed Dec 31 01:42:54 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   39C    P8              3W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

---
# Compile llamacpp natively first (don't skip this step!)
## Additional tools
```
sudo apt install ccache cmake build-essential libcurl4-openssl-dev
```

## Setup env variables
Based on your driver versions, `cuda` in particular, set the environment variables for your user properly in your `~/.bashrc`.
With Cuda v13.1, you should have something like this:

```
# CUDA 13.1 & GCC 12 Fix for ARM64
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-13.1
export CC=gcc-12
export CXX=g++-12
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
```

Then, relogin or load the file to make the changes effective.
```
$ . ~/.bashrc
```

## Clone repo (with submodue)
Clone this repository with `llama.cpp` submodule.
```
$ git clone --recurse-submodules https://github.com/cslev/llamacpp-cuda-arm64-docker.git
``` 
In case you cloned without the submodule, you can get it after checkout and changing the directory
```
$ cd llamacpp-cuda-arm64-docker
$ git submodule update --init --recursive
```

## Compile llama.cpp
```
$ cd llama.cpp
$ cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
$ cmake --build build --config Release -j$(nproc)
```

# Build Docker image
Let's build our multistage docker image. This might take some time, at least the same as building llama.cpp natively.
```
$ sudo docker build -t cslev/llamacpp-cuda-arm64 .
```

# Get models
Let's create a quick python environment, install `huggingface-hub` library, and pull some models. 
We are going to test it with a small vision model so that vision capabilities/image uploads can be verified too.

## Install python venv
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U "huggingface_hub"
```
## Download a small vision model
Download the model (GGUF) file first
```
$ hf download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF   Qwen2.5-VL-3B-Instruct-Q8_0.gguf  --local-dir ./models/
```
Download the mmproj file for Image input
```
$ hf download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF   mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf --local-dir ./models/
```

# Edit models.ini
Now, we need to edit the model description file `models.ini`, which can list all models that we downloaded and want to be served via llama.cpp.
```
$ nano models/models.ini
```
Then, modify and add new entries as per your requirements. Here, we just set up the model we just downloaded.
```
# Settings for Qwen2.5-VL
[Qwen2.5-VL-3B-instruct]
model = /models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf
mmproj = /models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
```

# Deploy
Using our `docker-compose.yml` file, we can easily deploy our freshly build llamacpp container.
```
$ sudo docker-compose up
```

After the stack is brought up, navigate to http://localhost:3000, and you can see have a full fledged llamacpp with CUDA support running on your ARM64 system within an isolated docker container.

<p align="center">
  <img src="assets/llamacpp_running.png" alt="llama.cpp running with CUDA">
</p>

---
After selecting a model, let's wait a few seconds till it is fully loaded and the Images attachment entry becomes active. Select a random image, and ask our small vision language model to describe the image.
As can be seen the performance is great, we got the response quite fast in seconds and the token/s metric also suggests the GPU is working effectively (in our DGX Spark - otherwise, with a CPU only approach it would be only ~10 token/s)

<p align="center">
  <img src="assets/llamacpp_describe_image.png" alt="llama.cpp describing an image">
</p>

