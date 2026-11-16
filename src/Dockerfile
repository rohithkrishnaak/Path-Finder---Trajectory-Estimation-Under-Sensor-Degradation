# Reproducible run of Path Finder (Python only, as required by the Track 3 FAQ).
#   docker build -t pathfinder -f src/Dockerfile src
#   docker run --rm --cpus=1 -v "$PWD/data:/data" pathfinder /data/input.zip -o /data/estimated_trajectory.txt
FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=1.24"
WORKDIR /app
COPY main.py README.md ./
ENTRYPOINT ["python", "main.py"]
