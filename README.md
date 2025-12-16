# 🚀 MLOps Assignment 2: End-to-End Machine Learning Pipeline

## 📌 Project Overview

This project implements a **complete end-to-end MLOps pipeline** for predicting housing prices using the **California Housing dataset**.
The system automates **model training, evaluation, and deployment** using industry-standard MLOps tools such as **Apache Airflow, FastAPI, and Docker**.

The trained machine learning model is orchestrated via Airflow and exposed through a **RESTful API** for real-time inference, ensuring **reproducibility, scalability, and modular deployment**.

---

## 🏗️ System Architecture

The solution consists of **two primary Dockerized services**:

### 🔄 Airflow Service (Training Pipeline)

Responsible for orchestrating the ML workflow:

1. **Install Dependencies** – Ensures required Python packages are available.
2. **Data Loading** – Fetches the California Housing dataset using Scikit-Learn.
3. **Model Training** – Trains a Linear Regression model.
4. **Evaluation & Logging** – Calculates Mean Squared Error (MSE) and saves the trained model.

### 🌐 FastAPI Service (Inference Layer)

* Loads the trained `model.pkl`
* Exposes a `/predict` endpoint for real-time predictions
* Provides interactive API testing via **Swagger UI**

---

## 📂 Project Structure

```bash
mlops-assignment-2/
│
├── dags/
│   └── train_pipeline.py     # Airflow DAG (ETL + training logic)
│
├── api/
│   ├── main.py               # FastAPI application
│   └── Dockerfile            # API container build instructions
│
├── docker-compose.yaml       # Multi-container orchestration
├── requirements.txt          # API dependencies
├── model.pkl                 # Trained model (generated post-pipeline)
└── README.md                 # Project documentation
```

---

## ⚙️ Prerequisites

Ensure the following are installed on your system:

* **Docker Desktop** (running)
* **Git**
* Minimum **8GB RAM** recommended for smooth container execution

---

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-assignment-2.git
cd mlops-assignment-2
```

### 2️⃣ Build & Start Services

```bash
docker compose up -d --build
```

⏳ *First-time setup may take a few minutes as Docker images are downloaded.*

---

## 💻 Usage Guide

### 🟢 Part 1: Model Training (Airflow)

1. Open Airflow UI:
   👉 [http://localhost:8080](http://localhost:8080)

2. Login credentials:

   * **Username:** airflow
   * **Password:** airflow

3. Locate the DAG: **`housing_train_pipeline`**

4. Unpause the DAG and click **▶ Trigger DAG**

5. Wait until all tasks turn **dark green (Success)**

6. Check logs of `log_results_task` to view the **MSE score**

---

### 🔁 Part 2: Update the Model (Critical Step)

Since the model is trained **inside the Airflow container**, it must be copied to the host machine for the API to use.

Run after the pipeline completes:

```powershell
docker cp mlops-assignment-2-airflow-worker-1:/tmp/model.pkl ./model.pkl
```

Restart the API service:

```powershell
docker compose restart fastapi-app
```

---

### 🔮 Part 3: Making Predictions (FastAPI)

1. Open Swagger UI:
   👉 [http://localhost:8000/docs](http://localhost:8000/docs)

2. Select **POST /predict**

3. Click **Try it out → Execute**

4. View the prediction response:

```json
{
  "prediction": 4.526
}
```

---

## 🛠️ Troubleshooting

### ❌ Localhost Refused / Empty Response

* Ensure Docker Desktop is running
* Wait **30–60 seconds** after startup for services to initialize

### ❌ Model Copy Error

* Ensure the Airflow pipeline completed successfully
* Copy the model immediately after training finishes
* Re-run the DAG if needed

### ❌ API Internal Server Error

* Verify `model.pkl` is not empty (file size > 0 KB)
* Re-copy the model and restart the API

---

## 📝 Learning Outcomes

* **Orchestration:** Designed dependency-driven pipelines using Apache Airflow
* **Containerization:** Built and managed multi-service systems with Docker & Docker Compose
* **Model Serving:** Implemented a production-ready inference API with FastAPI
* **MLOps Best Practices:** Separation of training & serving environments, reproducibility, and automation

---

## 📌 Technologies Used

* Python
* Scikit-Learn
* Apache Airflow
* FastAPI
* Docker & Docker Compose
* Pydantic
* REST APIs

---

## 👨‍💻 Author

**Muhammad Hassan Tahir**
MLOps / Machine Learning Engineer
---
