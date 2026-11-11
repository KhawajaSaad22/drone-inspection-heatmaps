FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home inspector && mkdir -p /data /models /outputs && chown inspector /outputs
USER inspector
ENTRYPOINT ["drone-inspect"]
CMD ["--help"]

FROM base AS model
USER root
RUN pip install --no-cache-dir '.[model]'
USER inspector
