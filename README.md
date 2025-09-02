![phData Logo](phData.png "phData Logo")
# House Price Prediction API — Project_phData

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

Build & start
```bash
docker compose up --build -d
```

Verify services
```bash
docker compose ps
```

Hit the API
```bash
curl -s http://localhost:8000/predict_core/ -X POST -H 'Content-Type: application/json' -d '{"bedrooms":3,"bathrooms":1,"sqft_living":1180,"sqft_lot":5650,"floors":1,"sqft_above":1180,"sqft_basement":0,"zipcode":"98178"}'
```

Check health
```bash
curl -s http://localhost:8000/health/
```

---

## Requirements


* Docker & Docker Compose
* Python 3.10+ (local training)
* Conda

To activate the environment run: 
```bash
conda env create -f conda_environment.yml
```

```bash
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

- Port: API exposed on host port **8000** via NGINX.
- Service roles:
  - `web` = blue app
  - `web_green` = green app
 - Docker Compose reads variables from a `.env` file that you create by copying
   the supplied `.env.demo`.  The Compose file uses `${VAR}` placeholders,
   so values defined in `.env` override the defaults baked into
   `docker-compose.yml`.  A second file, `.env.demo`, contains only
   non‑sensitive values and is used by the demo scripts described later.

| Variable | Description | Default |
|---|---|---|
| `DJANGO_SETTINGS_MODULE` | Django settings module | `Project_phData.settings` |
| `SECRET_KEY` | Django secret key | `change-me` |
| `DEMOGRAPHICS_PATH` | Demographics CSV | `data/zipcode_demographics.csv` |
| `BLUE_MODEL_PATH` | Blue model pkl | `models/blue/model.pkl` |
| `BLUE_MODEL_FEATURES_PATH` | Blue features JSON | `models/blue/model_features.json` |
| `BLUE_MODEL_VERSION` | Blue version | `1.0` |
| `GREEN_MODEL_PATH` | Green model pkl | `models/green/model.pkl` |
| `GREEN_MODEL_FEATURES_PATH` | Green features JSON | `models/green/model_features.json` |
| `GREEN_MODEL_VERSION` | Green version | `2.0` |

Example `.env`
```dotenv
DJANGO_SETTINGS_MODULE=Project_phData.settings
SECRET_KEY=change-me
DEMOGRAPHICS_PATH=data/zipcode_demographics.csv
BLUE_MODEL_PATH=models/blue/model.pkl
BLUE_MODEL_FEATURES_PATH=models/blue/model_features.json
BLUE_MODEL_VERSION=1.0
GREEN_MODEL_PATH=models/green/model.pkl
GREEN_MODEL_FEATURES_PATH=models/green/model_features.json
GREEN_MODEL_VERSION=2.0
```

Demo mode (no secrets on disk)

To spin up the stack for a demo without writing any secrets to disk,
use the helper scripts included in the repository.  These scripts
generate a random `SECRET_KEY` for the lifetime of the containers and
read all other variables from `.env.demo`.

```bash
bash demo_up.sh
```

The command will start the containers in detached mode.  When
finished, tear everything down using the provided script:

```bash
bash demo_down.sh
```

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
ln -sfn model models/blue
```

```bash
ln -sfn model_V2 models/green
```

Inspect
```bash
ls -l models
```

### Train
#### Baseline (v1)
```bash
python scripts/create_model.py
```
#### Extended (v2): tries KNN/RF/XGBoost and saves the best
```bash
python scripts/create_model_V2.py
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

Errors return HTTP 400 with `{"error": "message"}`.

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

See services
```bash
docker compose ps
```

Resource snapshot
```bash
docker stats --no-stream
```

Logs
```bash
docker compose logs nginx | tail -n 100
```

```bash
docker compose logs web | tail -n 100
```

```bash
docker compose logs web_green | tail -n 100
```

Add a Gunicorn worker to a container
```bash
docker kill --signal=TTIN project_phdata-web-1
```

Remove a Gunicorn worker
```bash
docker kill --signal=TTOU project_phdata-web-1
```

Verify workers
```bash
docker top project_phdata-web-1 | grep gunicorn
```

Scale up
```bash
docker compose up -d --scale web=3 --scale web_green=2
```

Scale down
```bash
docker compose up -d --scale web=1 --scale web_green=1
```

---

## Monitoring

Live container stats
```bash
docker stats
```

Cloud options include APM tools and AWS CloudWatch.

---

## Autoscaling Options

Example Kubernetes HPA
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

Best practices: stateless services, resource requests/limits, health probes, and optional custom metrics.

---

## CI

GitHub Actions run tests on pushes and PRs. A deploy job can build and ship via SSH on the default branch.

High level steps
- Checkout repository
- Setup Python
- Install dependencies
- Run tests
- Build and deploy on default branch

Configure secrets such as `EC2_HOST`, `EC2_USER`, and `SSH_PRIVATE_KEY` for deployment.

---

## Troubleshooting

- HTTP 400 responses: check request fields and model artifacts
- High latency: add a worker or scale replicas
- Model load issues: confirm `MODEL_PATH` and `MODEL_FEATURES_PATH`

