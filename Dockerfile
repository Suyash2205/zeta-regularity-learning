# Container for reproducing the paper.
#   docker build -t zrl .
#   docker run --rm -v "$PWD/results:/zrl/results" zrl            # full run (about 2 hours, 64 GB RAM)
#   docker run --rm zrl smoke                                      # quick check (about 15 minutes, 16 GB RAM)
FROM python:3.11.2-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /zrl
COPY requirements.txt .
RUN python -m venv .venv && .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
COPY . .
ENV ZRL_SKIP_INSTALL=1
ENTRYPOINT ["./reproduce.sh"]
