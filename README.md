MLOps Assignment 2: End-to-End Machine Learning Pipeline
📌 Project Overview
This project implements a complete MLOps pipeline for predicting housing prices using the California Housing dataset. It automates the data ingestion and model training process using Apache Airflow and serves the trained model via a FastAPI REST interface. The entire system is containerized using Docker for easy deployment and reproducibility.

🏗️ Architecture
The project consists of two main services running in Docker containers:

Airflow Service: Orchestrates the ETL pipeline (Extract, Train, Log).

Task 1: Installs dependencies.

Task 2: Loads data from Scikit-Learn.

Task 3: Trains a Linear Regression model.

Task 4: Logs the MSE score and saves the model.

FastAPI Service: Loads the trained model and provides an HTTP endpoint (/predict) for real-time inference.

📂 Project Structure
Bash

mlops-assignment-2/
│
├── dags/
│   └── train_pipeline.py     # Airflow DAG definition (ETL logic)
│
├── api/
│   ├── main.py               # FastAPI application code
│   └── Dockerfile            # Instructions to build the API container
│
├── docker-compose.yaml       # Orchestration for Airflow + FastAPI
├── requirements.txt          # Python dependencies for the API
├── model.pkl                 # The trained model (generated after pipeline run)
└── README.md                 # Project documentation
⚙️ Prerequisites
Docker Desktop installed and running.

Git installed.

🚀 Setup & Installation
1. Clone the Repository
Bash

git clone https://github.com/YOUR_USERNAME/mlops-assignment-2.git
cd mlops-assignment-2
2. Build and Start Services
Run the following command to download images and start the containers:

Bash

docker compose up -d --build
Note: The first run may take a few minutes to download the Docker images.

💻 Usage Guide
Part 1: Training the Model (Airflow)
Open your browser and navigate to http://localhost:8080.

Login Credentials:

Username: airflow

Password: airflow

Find the DAG named housing_train_pipeline.

Unpause the DAG (toggle the switch to Blue) and click the Play (▶) button to trigger a run.

Wait for all tasks to turn Dark Green (Success).

Check the logs of the log_results_task to see the Mean Squared Error (MSE).

Part 2: Updating the Model (Critical Step)
Since the model is trained inside the Airflow container, you must copy it to your project folder so the API can use it. Run this command in your terminal after the pipeline finishes:

PowerShell

docker cp mlops-assignment-2-airflow-worker-1:/tmp/model.pkl ./model.pkl
Then, restart the API to load the new model:

PowerShell

docker compose restart fastapi-app
Part 3: Making Predictions (FastAPI)
Navigate to the Swagger UI: http://localhost:8000/docs.

Click on the POST /predict endpoint.

Click Try it out.

The request body will auto-fill with default values from the California Housing dataset.

Click Execute.

Scroll down to see the predicted price in the response body!

JSON

{
  "prediction": 4.526
}
🛠️ Troubleshooting
1. "Empty Response" or Localhost Refused

Ensure Docker Desktop is running.

Wait 30-60 seconds after docker compose up for the webserver to fully initialize.

2. Model Copy Error (Could not find file)

The pipeline might not have finished successfully. Check the Airflow UI graph to ensure the train_model task is green.

The temporary file inside the container is deleted if the container restarts. Run the pipeline again immediately before copying.

3. API returns "Internal Server Error"

The model.pkl file might be empty (0kb). Verify the file size on your local machine. If it is 0kb, repeat the copy step.

📝 Learning Outcomes
Orchestration: learned how to define dependency graphs in Python using Airflow.

Containerization: Gained experience writing Dockerfiles and using Docker Compose to manage multi-container applications.

Model Serving: Implemented a real-time inference API ensuring strict schema validation with Pydantic.
