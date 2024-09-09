# celeryconfig.py
broker_url = 'amqp://guest:guest@rabbitmq:5672//'
result_backend = 'rpc://'




# Task result expiration (in seconds)
result_expires = 3600

# Enable UTC for time-based tasks
enable_utc = True

# Optional: Custom settings for concurrency, retries, etc.
worker_concurrency = 4
task_always_eager = False
