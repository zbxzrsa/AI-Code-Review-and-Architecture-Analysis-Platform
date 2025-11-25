import os
from celery import Celery

# Use Redis as broker and backend for simplicity
BROKER = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("app", broker=BROKER, backend=RESULT_BACKEND)

celery_app.conf.update(
    task_acks_late=True,
    worker_max_tasks_per_child=100,
    task_time_limit=int(os.getenv("TASK_TIME_LIMIT", "900")),
    task_soft_time_limit=int(os.getenv("TASK_SOFT_TIME_LIMIT", "840")),
    task_serializer="json",
    accept_content=["json"],
)
