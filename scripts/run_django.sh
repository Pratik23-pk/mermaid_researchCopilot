#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../services/django_web"
../../venv/bin/python manage.py runserver 127.0.0.1:8000
