from sqlalchemy import inspect

from shared.db import engine 
from shared.models import Base 

def main() -> None:
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())

    print(f"Database URL: {engine.url}")
    print(f"Tables present ({len(tables)}):")
    for name in tables:
        print(f"    - {name}")


if __name__ == "__main__":
    main()