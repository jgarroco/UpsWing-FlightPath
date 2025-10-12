# Flight-Path Assessment System

## Tech Stack

- **Backend**: Python 3.11, FastAPI
- **Database**: MySQL 8.0
- **ORM**: SQLAlchemy (async)
- **Data Validation**: Pydantic
- **Containerization**: Docker, Docker Compose
- **Database Migrations**: Alembic

## Prerequisites

- Docker Desktop
- Docker Compose

## Installation & Setup

1. Ensure Docker Desktop is installed and running
2. Clone the repository:

   ```bash
   git clone <repository-url>
   cd flight-path
   ```

## Running the Application

1. Build and start the services using Docker Compose:

   ```bash
   docker compose up --build -d
   ```

2. Alternatively, if images are already built:

   ```bash
   docker compose up -d
   ```

3. The application will be available at `http://localhost:8000`
4. The API documentation will be available at `http://localhost:8000/docs`

## Database Migrations

The application uses Alembic for database migrations:

### Applying Migrations

```bash
docker compose exec api alembic upgrade head
```

## Seeding Data

To seed initial data for assessments and questions:

```bash
docker compose exec api python -m scripts.seed_db
```

## Project Structure

```
flight-path/
├── app/
│   ├── application/      # Use case interactors and DTOs
│   ├── domain/          # Core business entities and rules
│   ├── infrastructure/  # Database persistence and external services
│   ├── presentation/    # API endpoints and request/response schemas
│   └── setup/          # Configuration and dependency injection
├── alembic/            # Database migration scripts
├── scripts/            # Utility scripts
├── Dockerfile          # Container build instructions
├── docker-compose.yml  # Multi-container orchestration
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
└── alembic.ini       # Migration configuration
```

## Architecture

The application follows a Clean Architecture pattern with the following layers:

- **Presentation Layer**: FastAPI routers handle HTTP requests and responses
- **Application Layer**: Contains use case interactors that orchestrate business operations as well as DTOs for properly handling model values
- **Domain Layer**: Core business entities and rules, including Computerized Adaptive Testing logic
- **Infrastructure Layer**: Database persistence, external service adapters, and configuration

## API Documentation

FastAPI provides automatic interactive API documentation at:

- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

## Database Access with MySQL Workbench

To connect to the database using MySQL Workbench

Create a new connection with the following settings:

- **Connection Name**: flight-path-db
- **Hostname**: localhost
- **Port**: 3308
- **Username**: wasd
- **Password**: Use the `MYSQL_PASSWORD` value from your `.env` file (default is "password")

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Ensure MySQL container is running (`docker compose ps`)
2. **Port Already in Use**: Check if port 8000 or 3308 is already occupied
3. **Migration Errors**: Run migrations manually with `docker compose exec api alembic upgrade head`
4. **Permission Errors**: Ensure Docker has necessary permissions to access the project directory

### Resetting the Database

To reset the database completely:

```bash
docker compose down -v
docker compose up -d
```
