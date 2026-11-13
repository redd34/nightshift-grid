# MCP server image (stdio). This is what MCP directories such as Glama build to check that the server
# starts and answers introspection; it doesn't run a grid. Point it at a real coordinator with
# NIGHTSHIFT_URL / NIGHTSHIFT_TOKEN_FILE and mount your documents under NIGHTSHIFT_MCP_ROOTS.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY *.py ./
RUN pip install --no-cache-dir . && useradd --create-home nightshift
USER nightshift
WORKDIR /home/nightshift
ENV NIGHTSHIFT_URL=http://host.docker.internal:8000
ENTRYPOINT ["nightshift-mcp"]
