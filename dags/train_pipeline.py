from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator  # Added this
from datetime import datetime, timedelta
import os

# Define paths
DATA_PATH = "/tmp/housing_data.csv"
MODEL_PATH = "/tmp/model.pkl"

def load_data(**kwargs):
    # --- MOVED IMPORTS INSIDE THE FUNCTION ---
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    # -----------------------------------------
    
    print("Loading data...")
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")

def train_model(**kwargs):
    # --- MOVED IMPORTS INSIDE THE FUNCTION ---
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    # -----------------------------------------

    print("Training model...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    kwargs['ti'].xcom_push(key='model_mse', value=mse)
    print(f"Model MSE: {mse}")

def log_results(**kwargs):
    ti = kwargs['ti']
    mse = ti.xcom_pull(key='model_mse', task_ids='train_model_task')
    print("---------------------------------------------------")
    print(f"PIPELINE FINISHED. Final Model MSE: {mse}")
    print("---------------------------------------------------")

default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'housing_train_pipeline',
    default_args=default_args,
    description='A simple MLOps training pipeline',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # 1. Install Libraries First
    t0 = BashOperator(
        task_id='install_requirements',
        bash_command='pip install pandas scikit-learn joblib'
    )

    t1 = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    t2 = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model,
        provide_context=True, 
    )

    t3 = PythonOperator(
        task_id='log_results_task',
        python_callable=log_results,
        provide_context=True,
    )

    # Order: Install -> Load -> Train -> Log
    t0 >> t1 >> t2 >> t3