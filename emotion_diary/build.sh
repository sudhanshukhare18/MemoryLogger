#!/usr/bin/env bash
# Exit on error
set -o errexit  

pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python manage.py collectstatic --noinput
python manage.py migrate
