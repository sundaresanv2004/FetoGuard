FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements
COPY requirements.txt .

# Install dependencies using uv
# --system installs into the system python environment, which is what we want in a container
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application
COPY . .

# Set entrypoint (can be overridden)
CMD ["/bin/bash"]
