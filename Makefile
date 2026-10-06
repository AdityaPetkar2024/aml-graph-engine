# AML Graph-Screening Engine: common commands. Run from the project root with the venv active.
PY ?= python

.PHONY: help pipeline ingest neo4j-import neo4j-check rules features train staleness up down logs api dashboard stream stream-full

help:            ## list commands
	@grep -E "^[a-z0-9-]+:.*##" Makefile | sed "s/:.*##/ -/"

# ---------- offline pipeline (stages 1-5) ----------
pipeline: ingest neo4j-import neo4j-check rules features train   ## run the whole offline pipeline

ingest:          ## stage 1: clean, tag patterns, split, write parquet + Neo4j CSVs
	$(PY) src/ingest.py

neo4j-import:    ## stage 2: bulk-import the graph (stops Neo4j while importing)
	docker compose stop neo4j
	mkdir -p neo4j_data
	docker run --rm --user "$$(id -u):$$(id -g)" -v "$$PWD/neo4j_data:/data" -v "$$PWD/data/processed:/import" \
		neo4j:5.26 neo4j-admin database import full --nodes=/import/nodes.csv \
		--relationships=/import/edges.csv --overwrite-destination neo4j
	docker compose up -d neo4j

neo4j-check:     ## stage 2: indexes + count check against the parquet
	$(PY) src/load_neo4j.py

rules:           ## stage 3: rule baseline in Cypher
	$(PY) src/patterns.py

features:        ## stage 4: account features per split
	$(PY) src/features.py

train:           ## stage 5: XGBoost variants, evaluation, SHAP, model selection
	$(PY) src/train.py

staleness:       ## graph-refresh schedule experiment (in-process, ~10 min)
	$(PY) src/staleness.py

# ---------- serving (stages 6-7) ----------
up:              ## start Neo4j, API and dashboard in Docker
	docker compose up -d --build

down:            ## stop everything
	docker compose down

logs:            ## follow API and dashboard logs
	docker compose logs -f api dashboard

api:             ## run the API on the host instead of Docker
	uvicorn api.main:app --port 8000 --log-level warning

dashboard:       ## run the dashboard on the host instead of Docker
	streamlit run dashboard/app.py

stream:          ## replay 5,000 test transactions one at a time (demo)
	$(PY) src/simulator.py --reset --n 5000 --rate 20

stream-full:     ## replay all 862,792 test transactions in micro-batches + evaluate
	$(PY) src/simulator.py --reset --all --batch-size 10000
