import subprocess
import time
from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.conf import settings

# Use psycopg 3
try:
    import psycopg
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Psycopg 3 not found. Install it with: pip install psycopg[binary]"
    )


class Command(BaseCommand):
    help = "Start Django app and ensure the TimescaleDB container is running"

    def handle(self, *args, **options):
        db_settings = settings.DATABASES['default']
        host = db_settings['HOST']
        port = db_settings['PORT']
        user = db_settings['USER']
        password = db_settings['PASSWORD']
        name = db_settings['NAME']

        self.stdout.write("Checking database connection...")

        max_retries = 10
        delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                # Psycopg 3 connection
                conn = psycopg.connect(
                    dbname=name,
                    user=user,
                    password=password,
                    host=host,
                    port=port
                )
                conn.close()
                self.stdout.write(self.style.SUCCESS("Database is up!"))
                break
            except psycopg.OperationalError:
                self.stdout.write(
                    f"Database not ready. Attempt {attempt + 1}/{max_retries}..."
                )
                if attempt == 0:
                    self.stdout.write("Starting Docker Compose services...")
                    subprocess.run(["docker-compose", "up", "-d"], check=True)
                time.sleep(delay)
        else:
            self.stdout.write(self.style.ERROR("Database is not reachable. Exiting."))
            return

        self.stdout.write("Running migrations...")
        call_command("migrate")

        self.stdout.write(self.style.SUCCESS("Startup complete. Django is ready!"))
