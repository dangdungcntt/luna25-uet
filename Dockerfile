FROM ghcr.io/astral-sh/uv@sha256:3a747e3f1b33fbc4fe8541fe49c881f5684370e53fcdc0e5e96c71d47ad1c699

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy files to the container
COPY --chown=user:user pyproject.toml .python-version uv.lock /opt/app/

RUN uv sync --locked

COPY --chown=user:user . /opt/app/

EXPOSE 8000

CMD ["uv", "run", "api.py"]
