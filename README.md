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
9. [Performing a Blue-Green Deployment](#performing-a-blue-green-deployment)
10. [Testing & Developer Tools](#testing--developer-tools)
11. [Operations: Diagnostics & Scaling](#operations-diagnostics--scaling)
12. [Monitoring](#monitoring)
13. [Autoscaling Options](#autoscaling-options)
14. [CI](#ci)
15. [Troubleshooting](#troubleshooting)

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
* Conda

To activate the environment run: 
```bash
conda env create -f conda_environment.yml

conda activate housing
```
---

## Project Structure

```

Project_phData/
├─ Project_phData/          # Django settings/urls/wsgi/asgi
├─ predictor/               # API app (views for /predict & /predict_core)
├─ data/                    # Source datasets
├─ models/                  # Trained model artifacts
│  ├─ blue/                 # Symlink blue -> model
│  ├─ green/                # Symlink green -> model_V2
│  ├─ model/                # Baseline model artifacts (v1)
│  └─ model_V2/             # Improved model artifacts (v2)
├─ scripts/                 # Train/evaluate/client scripts
├─ tests/                   # pytest API tests
├─ Dockerfile
├─ docker-compose.yml
├─ nginx.conf
├─ manage.py
├─ requirements.txt
├─ pytest.ini
├─ phData.png
└─conda_environment.yml
```

---

## Configuration & Environment

* **Ports:** NGINX exposes the API on host port **8000**.
* **Service roles:**

  * `web` = **blue** app (baseline model)
  * `web_green` = **green** app (candidate model)


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
#### Baseline (v1)
```bash
python Project_phData/scripts/create_model.py
```
#### Extended (v2): tries KNN/RF/XGBoost and saves the best
```bash
python Project_phData/scripts/create_model_V2.py
```

---

## Run Locally

```bash
docker compose up --build -d 
```
#### API served on http://localhost:8000/
Services started:

* `web` (blue) — serves v1 artifacts and returns `model_version="v1"`
* `web_green` (green) — serves v2 artifacts and returns `model_version="v2"`
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
  "model_version": "1.0",
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

## Performing a Blue-Green Deployment

1. **Start the existing stack.**  Bring up the current system as usual:
   ```bash
   docker-compose up --build -d
   ```
This starts the `web` (blue) and `nginx` services.  The API will be available at <http://localhost:8000>.
* `docker-compose.yml` defines **two web services** (`web` and `web_green`), each mounting different model artifacts and exposing a different `MODEL_VERSION` string.
* `nginx.conf` has a weighted upstream:

  ```nginx
  upstream django {
      server web:8000 weight=10;        # blue (v1)
      server web_green:8000 down;  # green (v2)
  }
  ```
* NGINX listens on host **:8000** and round-robins by the configured weights.

**Typical rollout:**

1. **Train and save the new model.**  Use `create_model_V2.py` to train an updated model.  Save the pickled model and
   feature list into `models/model_V2/` on the host.
   ```bash
   python scripts/create_model_V2.py
   ```
2. Start/update `web_green`:

   ```bash
   docker-compose up --build -d web_green
   ```
3. Verify with logs and sample requests.

   ```bash
   docker-compose logs -f web_green
   ```
4. Increase the green weight in `nginx.conf`, then reload:
      ```nginx
      upstream django {
          server web:8000 weight=9;        # blue (v1)
          server web_green:8000 weight=1;  # green (v2)
      }
      ```
    
   ```bash
   docker-compose exec nginx nginx -s reload
   ```

5. **Adjust traffic weights.**  To shift more traffic to the new model,
   edit `nginx.conf` and modify the weights in the `upstream django`
   block.  For example, to route half of the requests to each version:

   ```nginx
   upstream django {
       server web:8000 weight=5;
       server web_green:8000 weight=5;
   }
   ```

   After editing, reload NGINX inside the running container:

   ```bash
   docker-compose exec nginx nginx -s reload
   ```

   Repeat this process, gradually increasing the weight for `web_green`, until
   you are satisfied with the new model’s performance.

6. When satisfied, stop blue or remove it, reload NGINX.

   ```nginx
   upstream django {
       server web:8000 down;
       server web_green:8000 weight=10;
   }
   ```

   ```bash
   docker-compose exec nginx nginx -s reload
   ```
   ```bash
   docker-compose stop web
   docker-compose rm -f web
   ```
   
7. **Next model**
   Repoint the *inactive* color to the next artifacts (e.g., `model_V3`) and repeat.

```bash
ln -sfn model_V3 models/blue
```
---

## Testing & Developer Tools

* **pytest** from project root:

  ```bash
  pytest -v
  ```
* **Smoke test (single model):** `python scripts/test_api.py`
* **Compare blue vs green:** `python scripts/test_api_compare.py` — alternates POSTs and buckets by `model_version` (useful when upstream weights are equal).

---

## Operations: Diagnostics & Scaling

A quick checklist to see **what’s running**, **how it’s performing**, and **what to scale** without downtime.

### See what’s running

#### Check services
```bash
docker compose ps
```
#### Check workers
```bash
docker top project_phdata-web-1 | grep gunicorn
docker top project_phdata-web_green-1 | grep gunicorn

for c in project_phdata-web-1 project_phdata-web_green-1; do 
  echo "$c:"; p=$(docker top "$c" | grep -c gunicorn); echo "$((p-1)) workers"; done
```


### Resource snapshot

```bash
docker stats --no-stream project_phdata-web-1 project_phdata-web_green-1
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

#### Add a worker
```bash
docker kill --signal=TTIN project_phdata-web-1
```
#### Remove a worker
```bash
docker kill --signal=TTOU project_phdata-web-1
```
#### Verify
```bash
docker top project_phdata-web-1 | grep gunicorn
```
> Replace with `project_phdata-web_green-1` to tune the green service.

To make it **permanent**, set workers in `docker-compose.yml`:

```yaml
command: gunicorn Project_phData.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 60
```

### Horizontal scaling (replicas)

#### Scale up
```bash
docker compose up -d --scale web=3 --scale web_green=2
docker compose ps  
```

#### Scale down
```bash
docker compose up -d --scale web=1 --scale web_green=1
docker compose ps  
```
> **When to add workers:** If a single container is CPU‑bound but not maxing out system memory, increase Gunicorn `--workers`. This improves concurrency within one replica.
> **When to add replicas:** If you want resilience, need to spread load across hosts, or are already saturating one container even with more workers, scale replicas with `--scale`. Replicas sit behind NGINX, which balances requests across them.

---

## Monitoring
We can have a live view of CPU %, memory usage, network IO, and block IO for each container.
```bash
docker stats
```
### New Relic APM (Django + Gunicorn) - *Not implemented*

New Relic APM can provide deeper application-level visibility, including:  
- p95/p99 latency per endpoint  
- Error tracking and distributed traces  
- Custom dashboards  
- Alerts on key performance indicators  

### AWS CloudWatch (when running on AWS)

CloudWatch can be used to monitor both infrastructure and application performance:  

- **Infrastructure metrics:**  
  CPU, memory (via CloudWatch Agent), and network usage for ECS/EC2  

- **Application Load Balancer metrics:**  
  Latency percentiles (`TargetResponseTime` p50/p90/p95/p99) and 5xx error counts  

- **Alarms:**  
  Trigger on high p95 latency, elevated 5xx errors, or sustained high CPU.  
  Notifications can be sent via SNS → Slack/email.  

---


## Autoscaling Options 

This service is containerized and stateless (models mounted read‑only), which makes it autoscaling‑friendly.

### Kubernetes HPA (Horizontal Pod Autoscaler)

The **Horizontal Pod Autoscaler (HPA)** automatically adjusts the number of pod replicas in a Deployment (or StatefulSet/ReplicaSet) based on observed resource usage or custom metrics. This helps ensure the application scales up under load and scales down to save costs when idle.

#### How it works
- The **metrics server** collects resource usage data.  
- The HPA controller compares current usage against the defined target (e.g., average CPU utilization = 70%).  
- If usage is above target, more replicas are added; if below, replicas are reduced (within min/max limits).  
- Scaling actions are gradual, to avoid flapping (rapid up/down scaling).  

#### Example: CPU-based scaling
The following manifest configures autoscaling between 2 and 10 replicas based on average CPU utilization:

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

This means:  
- Start with at least 2 replicas.  
- Allow scaling up to 10 replicas.  
- Add/remove pods so that the **average CPU utilization across all pods stays around 70%**.  

#### Best practices & design notes
1. **Keep services stateless**  
   - Store state in external systems (databases, caches, object storage).  
   - This ensures pods can be killed/restarted without data loss.  

2. **Define resource requests & limits**  
   - Always set `resources.requests.cpu/memory` and `resources.limits.cpu/memory`.  
   - This allows the scheduler and HPA to make accurate scaling decisions.  

3. **Add health probes**  
   - `readinessProbe`: ensures traffic is only routed to pods that are ready.  
   - `livenessProbe`: restarts unhealthy pods automatically.  

4. **Use custom metrics (beyond CPU/memory)**  
   - CPU/memory are good defaults, but may not reflect user experience.  
   - For latency-aware scaling, expose metrics like **p95 response time** or **queue length** via Prometheus, then configure HPA with a **Prometheus Adapter**.  

5. **Test scaling thresholds**  
   - Run load tests to validate your min/max settings and target thresholds.  
   - Ensure scaling actions happen early enough to prevent request failures.  

---

## CI

Continuous Integration (CI) is set up using **GitHub Actions** to automatically run tests on every `push` and `pull_request`. This ensures code quality and catches errors early in the development cycle.

### Workflow Overview

- **Trigger events:**  
  Runs on all pushes and pull requests.  

- **Environment:**  
  Uses the latest Ubuntu runner with Python 3.9.  

- **Steps:**  
  1. **Checkout repository** – pulls the code into the runner.  
  2. **Set up Python** – installs Python 3.9.  
  3. **Install dependencies** – installs project requirements and `pytest-django`.
  4. **Run tests** – executes the test suite with environment variables (model paths, dataset paths, etc.).  


## Troubleshooting

* **HTTP 400 with `{ "error": ... }`** — Bad/missing fields (e.g., `zipcode`), non‑JSON body, or model failed to load. Check app logs and confirm model artifacts exist and match the expected `model_features.json` order.
* **High latency / timeouts** — Check CPU and p95; add a Gunicorn worker or scale replicas; investigate slow endpoints in APM.

---


