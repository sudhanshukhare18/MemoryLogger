#!/usr/bin/env bash
# Exit on error
set -o errexit  

pip install --upgrade pip

pip install torch==2.9.0 torchvision==0.19.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python manage.py collectstatic --noinput
python manage.py migrate
