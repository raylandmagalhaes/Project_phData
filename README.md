# House Price Prediction API — Project\_phData

A production‑ready example of serving ML predictions behind a Django + Gunicorn API with an NGINX reverse proxy. It includes:

* Model training scripts with versioning (baseline v1, v2+)
* HTTP prediction service (`/predict`, `/predict_core`)
* **Blue‑green deployment** via Docker Compose + weighted NGINX upstreams
* Automated tests and example client scripts
* Production ops: monitoring, scaling, and autoscaling options

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Configuration & Environment](#configuration--environment)
5. [Data & Feature Engineering](#data--feature-engineering)
6. [Model Versioning & Training](#model-versioning--training)
7. [Run Locally](#run-locally)
8. [API Reference](#api-reference)
9. [Blue‑Green Deployment](#blue-green-deployment)
10. [Testing & Developer Tools](#testing--developer-tools)
11. [Operations: Diagnostics & Scaling](#operations-diagnostics--scaling)
12. [Monitoring](#monitoring)
13. [Autoscaling Options](#autoscaling-options)
14. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1) Build & start
docker compose up --build -d

# 2) Verify services
docker compose ps

# 3) Hit the API
curl -s http://localhost:8000/predict_core -X POST -H 'Content-Type: application/json' \
  -d '{"bedrooms":3,"bathrooms":1,"sqft_living":1180,"sqft_lot":5650,"floors":1,
       "sqft_above":1180,"sqft_basement":0,"zipcode":"98178"}'
```

The API listens on `http://localhost:8000/` via NGINX.

---

## Requirements

* Docker & Docker Compose
* Python 3.10+ (local training)

---

## Project Structure

```
Project_phData/
├─ Project_phData/
│  ├─ Project_phData/         # Django settings/urls/wsgi/asgi
│  ├─ predictor/               # API app (views for /predict & /predict_core)
│  ├─ data/                    # Source datasets
│  ├─ models/                  # Trained model artifacts
│  ├─ scripts/                 # Train/evaluate/client scripts
│  ├─ tests/                   # pytest API tests
│  ├─ Dockerfile
│  ├─ docker-compose.yml
│  ├─ nginx.conf
│  ├─ manage.py
│  ├─ requirements.txt
│  └─ pytest.ini
├─ phData.png
└─ conda_environment.yml
```

---

## Configuration & Environment

* **Ports:** NGINX exposes the API on host port **8000**.
* **Service roles:**

  * `web` = **blue** app (baseline model)
  * `web_green` = **green** app (candidate model)
* **Model version string:** exposed in responses as `model_version`.

> Tip: Adjust Gunicorn workers at runtime with signals (see [Operations](#operations-diagnostics--scaling)).

---

## Data & Feature Engineering

* **Sales data:** `data/kc_house_data.csv` (bedrooms, bathrooms, sqft, etc.)
* **Demographics:** `data/zipcode_demographics.csv` automatically merged by `zipcode` inside the API; callers do **not** send demographic fields.
* **Feature variants:**

  * **v1 (baseline):** subset of raw sales + demographics; no date features.
  * **v2:** wider raw features (e.g., `waterfront`, `view`, `condition`, `grade`, `lat/long`), date‑derived features (`sale_year`, `sale_month`, `sale_dayofweek`, `sale_quarter`, seasonal sin/cos), and `home_age_at_sale`; trains with a **log‑price target** via `TransformedTargetRegressor` (learn on log(price), predict \$).
* Each trained version ships with:

  * `model.pkl` — fitted estimator (pipeline included)
  * `model_features.json` — exact inference order expected by the API

---

## Model Versioning & Training

### Versioned folders

* `models/model/`  → v1 artifacts
* `models/model_V2/` → v2 artifacts

### "Blue/Green" symlink strategy

Create role‑based symlinks so deployments point to **roles** instead of hardcoded version paths:

```bash
ln -sfn model     models/blue
ln -sfn model_V2  models/green
ls -l models   # blue -> model, green -> model_V2
```

Update the *inactive* color to promote a new version later (e.g., repoint `blue` to `model_V3`).

### Train

```bash
# Baseline (v1)
python Project_phData/scripts/create_model.py

# Extended (v2): tries KNN/RF/XGBoost and saves the best
python Project_phData/scripts/create_model_V2.py
```

---

## Run Locally

```bash
docker compose up --build -d
# API served on http://localhost:8000/
```

Services started:

* `web` (blue) — serves v1 artifacts and returns `model_version="1.0"`
* `web_green` (green) — serves v2 artifacts and returns `model_version="2.0"`
* `nginx` — reverse proxy with weighted upstream to blue/green

---

## API Reference

### Endpoints

* `POST /predict/` — full input schema (no demographics required). The service:

  1. Validates JSON
  2. Builds **date features** (uses "today" if `date` missing)
  3. Joins demographics by `zipcode`
  4. Reorders to `model_features.json`
  5. Returns prediction(s)

* `POST /predict_core/` — minimal baseline schema:

  ```
  bedrooms, bathrooms, sqft_living, sqft_lot, floors,
  sqft_above, sqft_basement, zipcode
  ```

### Example (`/predict`)

```json
{
  "bedrooms": 3,
  "bathrooms": 1,
  "sqft_living": 1180,
  "sqft_lot": 5650,
  "floors": 1,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 1180,
  "sqft_basement": 0,
  "yr_built": 1955,
  "yr_renovated": 0,
  "zipcode": "98178",
  "lat": 47.5112,
  "long": -122.257,
  "sqft_living15": 1340,
  "sqft_lot15": 5650,
  "date": "20141013T000000"
}
```

**Response**

```json
{
  "model_version": "2.0",
  "features": [{ "...": "features used for prediction (not including engineered/demographic)"}],
  "prediction": [404549.0]
}
```

### Errors

* Invalid JSON / missing required fields (e.g., `zipcode`) → HTTP 400 with body:

  ```json
  { "error": "human-readable message…" }
  ```
* If the service fails to load a model at startup, requests return HTTP 400 with a helpful `error` message.

---

## Blue‑Green Deployment

Safely run old & new model versions side‑by‑side and shift real traffic with zero downtime.

1. **Start stack**

```bash
docker-compose up --build -d
# API at http://localhost:8000
```

`docker-compose.yml` defines two web services (blue/green). `nginx.conf` routes to them with weights:

```nginx
upstream django {
  server web:8000 weight=10;      # blue (v1)
  server web_green:8000 down;     # green (v2)
}
```

2. **Bring up green**

```bash
python Project_phData/scripts/create_model_V2.py
docker-compose up --build -d web_green
```

Check logs & sample requests:

```bash
docker-compose logs -f web_green
```

3. **Shift traffic gradually**
   Adjust weights in `nginx.conf` and reload inside the running proxy:

```nginx
upstream django {
  server web:8000 weight=9;
  server web_green:8000 weight=1;
}
```

```bash
docker-compose exec nginx nginx -s reload
```

Increase the green weight as confidence grows. For 50/50:

```nginx
upstream django {
  server web:8000 weight=5;
  server web_green:8000 weight=5;
}
```

4. **Promote green**
   When satisfied, disable blue and keep green:

```nginx
upstream django {
  server web:8000 down;
  server web_green:8000 weight=10;
}
```

```bash
docker-compose exec nginx nginx -s reload
docker-compose stop web && docker-compose rm -f web
```

5. **Next model**
   Repoint the *inactive* color to the next artifacts (e.g., `model_V3`) and repeat.

---

## Testing & Developer Tools

* **pytest** from project root:

  ```bash
  pytest -q
  ```
* **Smoke test (single model):** `python scripts/test_api.py`
* **Compare blue vs green:** `python scripts/test_api_compare.py` — alternates POSTs and buckets by `model_version` (useful when upstream weights are equal).

---

## Operations: Diagnostics & Scaling

A quick checklist to see **what’s running**, **how it’s performing**, and **what to scale** without downtime.

### See what’s running

```bash
docker compose ps
# Inspect Gunicorn processes (1 master + N workers)
docker top project_phdata-web-1 | grep gunicorn
docker top project_phdata-web_green-1 | grep gunicorn

# Quick worker counts (subtract 1 for master)
for c in project_phdata-web-1 project_phdata-web_green-1; do 
  echo "$c:"; p=$(docker top "$c" | grep -c gunicorn); echo "$((p-1)) workers"; done
```

### Resource snapshot

```bash
docker stats --no-stream project_phdata-web-1 project_phdata-web_green-1 nginx
```

**Rules of thumb**

* CPU **> 70–80%** for minutes + rising p95 → add a worker or replica
* CPU **< 30%** with low p95 → remove a worker/replica

### Errors & health

```bash
docker compose logs nginx | tail -n 100
docker compose logs web | tail -n 100
docker compose logs web_green | tail -n 100
```

### Adjust Gunicorn workers live (no restart)

```bash
# Add a worker
docker kill --signal=TTIN project_phdata-web-1
# Remove a worker
docker kill --signal=TTOU project_phdata-web-1
# Verify
docker top project_phdata-web-1 | grep gunicorn
```

> Replace with `project_phdata-web_green-1` to tune the green service.

To make it **permanent**, set workers in `docker-compose.yml`:

```yaml
command: gunicorn Project_phData.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 60
```

### Horizontal scaling (replicas)

```bash
# Scale up
docker compose up -d --scale web=3 --scale web_green=2
# Scale down
docker compose up -d --scale web=1 --scale web_green=1
```

> **When to add workers:** If a single container is CPU‑bound but not maxing out system memory, increase Gunicorn `--workers`. This improves concurrency within one replica.
> **When to add replicas:** If you want resilience, need to spread load across hosts, or are already saturating one container even with more workers, scale replicas with `--scale`. Replicas sit behind NGINX, which balances requests across them.

---

## Monitoring

### New Relic APM (Django + Gunicorn)

**What you get:** p95/p99 latency per endpoint, error tracking, traces, dashboards, and alerts.

Install and run under New Relic:

```bash
pip install newrelic
newrelic-admin generate-config YOUR_LICENSE_KEY newrelic.ini
NEW_RELIC_CONFIG_FILE=newrelic.ini \
NEW_RELIC_ENVIRONMENT=production \
newrelic-admin run-program \
  gunicorn Project_phData.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 60
```

Docker Compose example (per service):

```yaml
services:
  web:
    environment:
      NEW_RELIC_LICENSE_KEY: "${NEW_RELIC_LICENSE_KEY}"
      NEW_RELIC_APP_NAME: "House Price API (blue)"
      NEW_RELIC_ENVIRONMENT: "production"
      NEW_RELIC_LOG: "stdout"
    volumes:
      - ./newrelic.ini:/app/newrelic.ini:ro
    command: >
      newrelic-admin run-program
      gunicorn Project_phData.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 60

  web_green:
    environment:
      NEW_RELIC_LICENSE_KEY: "${NEW_RELIC_LICENSE_KEY}"
      NEW_RELIC_APP_NAME: "House Price API (green)"
      NEW_RELIC_ENVIRONMENT: "production"
      NEW_RELIC_LOG: "stdout"
    volumes:
      - ./newrelic.ini:/app/newrelic.ini:ro
    command: >
      newrelic-admin run-program
      gunicorn Project_phData.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 60
```

### AWS CloudWatch (if on AWS)

* Infra metrics: CPU, memory (via CloudWatch Agent), network for ECS/EC2
* ALB latency percentiles: `TargetResponseTime` p50/p90/p95/p99 + 5xx counts
* Alarms: p95 high, 5xx high, CPU high → notify via SNS/Slack/email
* Optional: push custom app latency from Django middleware with `PutMetricData`

---

## Autoscaling Options

This service is containerized and stateless (models mounted read‑only), which makes it autoscaling‑friendly.

### A) Manual (Docker Compose + NGINX)

```bash
# Scale up
docker compose up -d --scale web=3 --scale web_green=2
# Scale down
docker compose up -d --scale web=1 --scale web_green=1
```

> See also: [Horizontal scaling (replicas)](#horizontal-scaling-replicas) for the same command in the Ops section.

### B) Kubernetes HPA

Attach an HPA to your Deployment to adjust replicas automatically based on CPU or custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: house-price-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: house-price-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Design notes:** keep the service stateless, add `readinessProbe`, set requests/limits, and expose a custom metric for latency‑aware scaling (e.g., p95 via Prometheus Adapter).

### C) AWS ECS/Fargate Service Auto Scaling

Use CloudWatch target tracking on CPU or ALB metrics (`RequestCountPerTarget`, `TargetResponseTime (p95)`) to add/remove tasks; ECS drains tasks gracefully during scale actions.

### D) Safety & Rollback

Use the built‑in blue‑green pattern, keep graceful timeouts, prefer 429 on overload, and alert on p95/error rates/CPU.

---

## Troubleshooting

* **HTTP 400 with `{ "error": ... }`** — Bad/missing fields (e.g., `zipcode`), non‑JSON body, or model failed to load. Check app logs and confirm model artifacts exist and match the expected `model_features.json` order.
* **Green receives no traffic** — Ensure `web_green` is running and not marked `down` in `nginx.conf`; reload NGINX after edits.
* **High latency / timeouts** — Check CPU and p95; add a Gunicorn worker or scale replicas; investigate slow endpoints in APM.

---


