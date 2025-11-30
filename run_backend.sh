#!/bin/bash

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip (good practice)
pip install --upgrade pip

# 4. Install requirements
echo "Installing dependencies..."
pip install -r backend/requirements.txt

# 5. Run the application
echo "Starting server..."
# We need to set PYTHONPATH so python can find the 'ml' module from inside 'backend'
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 backend/main.py
