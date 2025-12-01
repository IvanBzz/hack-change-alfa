#!/bin/bash

# Функция для показа справки
show_help() {
    echo "Usage: ./run_backend.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  -r, --retrain        Retrain model and start server"
    echo "  --retrain-only       Retrain model only, don't start server"
    echo "  -h, --help           Show this help message"
    echo "  (no arguments)       Start server normally"
    exit 0
}

# Функция для переобучения модели
retrain_model() {
    echo "=== Starting full retraining pipeline ==="
    
    echo "1. Data preprocessing..."
    python ml/01_data_preprocessing.py
    
    echo "2. Model training..."
    python ml/02_model_training.py
    
    echo "3. SHAP analysis..."
    python ml/03_shap_analysis.py
    
    echo "4. Generating recommendations..."
    python ml/04_generate_recommendations.py
    
    echo "=== Retraining completed successfully ==="
}

# 1. First check command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
elif [ -n "$1" ] && [ "$1" != "--retrain" ] && [ "$1" != "-r" ] && [ "$1" != "--retrain-only" ]; then
    echo "Unknown option: $1"
    echo "Use ./run_backend.sh --help for usage information"
    exit 1
fi

# 2. Now setup environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
echo "Installing dependencies..."
pip install -r backend/requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Now handle retrain options
if [ "$1" = "--retrain" ] || [ "$1" = "-r" ]; then
    retrain_model
    echo "Starting server after retraining..."
elif [ "$1" = "--retrain-only" ]; then
    retrain_model
    echo "Retraining completed. Server not started."
    exit 0
fi

# 4. Run the application
echo "Starting server..."
python3 backend/main.py