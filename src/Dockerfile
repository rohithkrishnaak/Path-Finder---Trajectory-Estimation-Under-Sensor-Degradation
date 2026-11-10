# Reproducible Linux build + run of the Path Finder runtime.
#   docker build -t pathfinder -f src/Dockerfile src
#   docker run --rm --cpus=1 -v "$PWD/data:/data" pathfinder /data/input.zip -o /data/estimated_trajectory.txt
FROM gcc:13 AS build
WORKDIR /src
COPY . .
RUN sh build.sh

FROM debian:bookworm-slim
COPY --from=build /src/pathfinder /usr/local/bin/pathfinder
ENTRYPOINT ["pathfinder"]
