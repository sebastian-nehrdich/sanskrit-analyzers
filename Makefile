IMAGE_NAME ?= sanskrit-byt5-analyzer
PROJECT_DEBUG ?= sanskrit-byt5-analyzer-debug
PROJECT_PROD ?= sanskrit-byt5-analyzer-prod
COMPOSE ?= docker compose

.PHONY: build run-debug run-prod stop-debug stop-prod load-dictionaries load-dictionaries-debug elasticsearch-up

build:
	IMAGE_NAME=$(IMAGE_NAME) $(COMPOSE) build

run-debug: stop-debug
	HOST_APP_PORT=3417 APP_PORT=3417 UVICORN_RELOAD=1 IMAGE_NAME=$(IMAGE_NAME) COMPOSE_PROJECT_NAME=$(PROJECT_DEBUG) $(COMPOSE) up --build -d app elasticsearch

run-prod: stop-prod
	HOST_APP_PORT=3415 APP_PORT=3415 IMAGE_NAME=$(IMAGE_NAME) COMPOSE_PROJECT_NAME=$(PROJECT_PROD) $(COMPOSE) up --build -d app elasticsearch

stop-debug:
	-@COMPOSE_PROJECT_NAME=$(PROJECT_DEBUG) $(COMPOSE) down >/dev/null 2>&1 || true

stop-prod:
	-@COMPOSE_PROJECT_NAME=$(PROJECT_PROD) $(COMPOSE) down >/dev/null 2>&1 || true

elasticsearch-up:
	IMAGE_NAME=$(IMAGE_NAME) COMPOSE_PROJECT_NAME=$(PROJECT_PROD) $(COMPOSE) up -d elasticsearch

load-dictionaries: build elasticsearch-up
	IMAGE_NAME=$(IMAGE_NAME) COMPOSE_PROJECT_NAME=$(PROJECT_PROD) $(COMPOSE) run --rm app python load_dictionaries.py

load-dictionaries-debug: build
	IMAGE_NAME=$(IMAGE_NAME) COMPOSE_PROJECT_NAME=$(PROJECT_DEBUG) $(COMPOSE) up -d elasticsearch
	IMAGE_NAME=$(IMAGE_NAME) COMPOSE_PROJECT_NAME=$(PROJECT_DEBUG) $(COMPOSE) run --rm app python load_dictionaries.py
