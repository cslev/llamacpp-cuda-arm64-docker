# Use NVIDIA's official CUDA devel image for ARM64
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libcurl4-openssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the llama.cpp repository
COPY llama.cpp/ .



# Build llama.cpp with CUDA and cURL support from scratch
# remove any existing build directory to ensure a clean build
RUN rm -rf build && \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
 #   -DLLAMA_FATAL_WARNINGS=ON \
 #   -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
 #   -DCMAKE_CUDA_ARCHITECTURES="72" \
    . && \
    cmake --build build --config Release -j $(nproc)

# Final stage: Create a slim runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libcurl4 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the compiled binaries from the build stage
COPY --from=build /app/build/bin/llama-server /app/llama-server
COPY --from=build /app/build/bin/*.so* /app/

# Create a directory for models to be mounted
RUN mkdir /models

# Set environment variables for CUDA
#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH

EXPOSE 8033

# The entrypoint allows you to pass any model/params at runtime
ENTRYPOINT ["/app/llama-server"]

