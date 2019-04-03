#!/usr/bin/env bash
export FLASK_APP=data_server.py

gunicorn -w 1 -b 0.0.0.0:5000 data_server:app