FROM ghcr.io/openadmet/openadmet-models:main

USER root
RUN micromamba run -n base python -m pip install --no-cache-dir fastapi uvicorn
USER mambauser

WORKDIR /app
COPY --chown=mambauser:mambauser app.py /app/app.py

EXPOSE 8000
CMD ["micromamba", "run", "-n", "base", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
