# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements-service.lock ./requirements-service.lock
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-service.lock && pip check

FROM dependencies AS source
COPY pyproject.toml ./
COPY Dockerfile ./
COPY src ./src
RUN pip install --no-build-isolation --no-deps -e .

FROM source AS test
COPY tests ./tests
COPY configs ./configs
COPY tools ./tools
COPY results/foolad-hourly-a100 ./results/foolad-hourly-a100
COPY deploy ./deploy
COPY compose.yaml ./compose.yaml

FROM source AS bundle
COPY results/foolad-hourly-a100 /research
RUN marketforecast-service bundle --source /research --output /bundle && \
    marketforecast-service verify --bundle /bundle

FROM source AS production
RUN groupadd --gid 10001 forecast && useradd --uid 10001 --gid forecast --create-home forecast && \
    mkdir -p /state /registry /backups /mlflow && chown forecast:forecast /state /registry /backups /mlflow
COPY --from=bundle --chown=forecast:forecast /bundle /bundle
USER forecast
ENV MARKETFORECAST_BUNDLE=/bundle MARKETFORECAST_STATE=/state/audit.sqlite \
    MARKETFORECAST_REGISTRATION=/registry/registration.json
EXPOSE 8000
STOPSIGNAL SIGTERM
CMD ["marketforecast-service", "serve"]
