MLOps Project: Iris Classification API
This project demonstrates a minimal but complete MLOps pipeline for building, tracking, packaging, deploying, and monitoring a machine learning model for the Iris dataset.

Course Event: "Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices"

Technologies Used
Version Control: Git & GitHub

Experiment Tracking: MLflow

API Framework: FastAPI with Pydantic

Containerization: Docker & Docker Compose

CI/CD: GitHub Actions

Logging & Monitoring: Python logging module, Prometheus

Project Structure
.
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── data/raw/               # Raw dataset storage
├── mlruns/                 # MLflow experiment tracking artifacts and database
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data/               # Data processing scripts
│   ├── training/           # Model training scripts
│   └── utils/              # Utility modules (e.g., logging)
├── tests/                  # Pytest tests
├── .dockerignore           # Files to ignore in Docker build
├── .flake8                 # Configuration for the flake8 linter
├── .gitignore              # Files to ignore in Git
├── docker-compose.yml      # Docker Compose configuration for local development
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── summary.md              # Project summary document

How to Run the Project (Recommended Method) 🚀
This project uses Docker Compose to create a consistent and portable environment for both training and serving the model. This is the simplest and most reliable way to run it.

1. Prerequisites
You must have Docker and Docker Compose installed.

You have cloned this repository.

2. Prepare the Initial Dataset
This step only needs to be run once. It downloads the Iris dataset and places it in the data/raw directory.

python src/data/make_dataset.py

3. Run the MLOps Workflow with Docker Compose
Step A: Train the Model
This command builds the Docker image and runs the training service. It executes the training script inside a container, ensuring that the resulting MLflow artifacts (mlruns directory) are created in a clean, Linux-native environment and saved to your local machine.

docker-compose run --rm training

Step B: Run the API Server
Once training is complete, start the API service. This command starts the FastAPI server and mounts the mlruns directory created in the previous step.

docker-compose up api

Your API is now running and can be accessed in your browser at http://localhost:8000/docs.

Step C: Stop the Services
When you are finished, press Ctrl+C in the terminal where the API is running. To fully stop and remove the containers and network, run:

docker-compose down

<details>
<summary><strong>Alternative: Manual Local Run Without Docker Compose</strong></summary>

This method allows you to run each component of the project manually on your local machine.

Local Setup:

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

Generate Raw Data:

python src/data/make_dataset.py

Run Experiments and Register Model:
In one terminal, start the MLflow UI:

mlflow ui --backend-store-uri ./mlruns

In a second terminal, run the training script:

python src/training/train.py

Open http://127.0.0.1:5000 to view experiments.

Run the API Locally:

uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Access the API docs at http://localhost:8000/docs.

</details>

CI/CD Pipeline
The pipeline is defined in .github/workflows/ci-cd.yml and performs the following on push to main:

Lint & Test: Runs flake8 and pytest to ensure code quality.

Build & Push: Builds the Docker image and pushes it to Docker Hub.

Secrets Required in GitHub:

DOCKERHUB_USERNAME: Your Docker Hub username.

DOCKERHUB_TOKEN: A Docker Hub access token with Read, Write, Delete permissions.

API Endpoints
GET /: Health check.

GET /docs: Interactive API documentation (Swagger UI).

POST /predict: Makes a prediction.

GET /metrics: Exposes Prometheus metrics.