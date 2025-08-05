# MLOps Project: Iris Classification API

This project demonstrates a minimal but complete MLOps pipeline for building, tracking, packaging, deploying, and monitoring a machine learning model for the Iris dataset.

**Course Event:** "Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices"

## Technologies Used
- **Version Control:** Git & GitHub
- **Experiment Tracking:** MLflow
- **API Framework:** FastAPI
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Logging & Monitoring:** Python `logging` module, Prometheus

---

## Project Structure
```
.
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── data/raw/               # Raw dataset storage
├── logs/                   # API logs
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data/               # Data processing scripts
│   ├── training/           # Model training scripts
│   └── utils/              # Utility modules (e.g., logging)
├── tests/                  # Pytest tests
├── .dockerignore           # Files to ignore in Docker build
├── .gitignore              # Files to ignore in Git
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── summary.md              # Project summary document
```

---

## How to Run the Pipeline

### 1. Prerequisites
- Python 3.9+
- Docker
- A GitHub account
- A Docker Hub account

### 2. Local Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-iris-project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Steps
**Step 1: Generate Raw Data**
```bash
python src/data/make_dataset.py
```

**Step 2: Run Experiments and Register Model**
```bash
# In one terminal, start the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# In a second terminal, run the training script
python src/training/train.py
```
> Open `http://127.0.0.1:5000` to view experiments. The best model will be automatically registered and moved to "Production".

**Step 3: Run the API Locally**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
> Access the API docs at `http://127.0.0.1:8000/docs`.

**Step 4: Run with Docker**
```bash
# Build the Docker image
docker build -t your-dockerhub-username/mlops-iris-api .

# Run the container
docker run -p 8000:8000 your-dockerhub-username/mlops-iris-api
```
---

## CI/CD Pipeline
The pipeline is defined in `.github/workflows/ci-cd.yml` and performs the following on push to `main`:
1.  **Lint & Test:** Runs `flake8` and `pytest` to ensure code quality.
2.  **Build & Push:** Builds the Docker image and pushes it to Docker Hub at `your-dockerhub-username/mlops-iris-api:latest`.

**Secrets Required:**
- `DOCKERHUB_USERNAME`: Your Docker Hub username.
- `DOCKERHUB_TOKEN`: A Docker Hub access token.

## API Endpoints
- `GET /`: Health check.
- `GET /docs`: Interactive API documentation (Swagger UI).
- `POST /predict`: Makes a prediction.
- `GET /metrics`: Exposes Prometheus metrics.