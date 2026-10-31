# CPU image for the API and the Streamlit app. Models and the index are mounted, not baked in.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU build of PyTorch first (small); the project's model is small enough to run on a CPU for a demo.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data
COPY .streamlit ./.streamlit
COPY api.py app.py ./

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
