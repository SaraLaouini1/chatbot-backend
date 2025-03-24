from __future__ import with_statement
import os
from logging.config import fileConfig
from flask import current_app
from alembic import context

config = context.config
fileConfig(config.config_file_name)
target_metadata = current_app.extensions['migrate'].db.Model.metadata

def run_migrations_online():
    connectable = current_app.extensions['migrate'].db.engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True
        )

        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
