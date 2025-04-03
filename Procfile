web: if [ ! -d "migrations" ]; then flask db init; fi && flask db migrate -m "Auto migration" && flask db upgrade && gunicorn app:app --worker-tmp-dir /dev/shm --bind 0.0.0.0:${PORT}
