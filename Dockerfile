# Backend image for Hugging Face Spaces (Docker SDK). Serves the FastAPI app on 7860.
FROM python:3.11-slim

# HF Spaces runs the container as a non-root user (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install CPU-only torch first so the rest of requirements.txt doesn't pull the
# multi-GB CUDA build (HF free Spaces are CPU-only).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# App code + seeded vectorstore (demo data). Secrets come from Space settings, not files.
COPY --chown=user app ./app
COPY --chown=user vectorstores ./vectorstores

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
