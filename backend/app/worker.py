import os
from celery import Celery

BROKER = os.getenv("CELERY_BROKER_URL", "pyamqp://guest@localhost//")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

celery_app = Celery("app", broker=BROKER, backend=RESULT_BACKEND)

celery_app.conf.update(
    task_acks_late=True,
    worker_max_tasks_per_child=100,
    task_time_limit=int(os.getenv("TASK_TIME_LIMIT", "900")),
    task_soft_time_limit=int(os.getenv("TASK_SOFT_TIME_LIMIT", "840")),
    task_serializer="json",
    accept_content=["json"],
)
