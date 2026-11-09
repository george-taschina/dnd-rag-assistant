.PHONY: install app cli test test-unit ci eval-retrieval eval-batch eval-judge eval-metrics eval-tune eval-logs label-app docker-build docker-up

install:
	uv sync

app:
	uv run streamlit run app.py

cli:
	uv run python main.py "How does an opportunity attack work? What triggers it and what does it cost?"

test:
	uv run pytest tests/ trace_replay/ -v

test-unit:
	uv run pytest trace_replay/ -v

eval-retrieval:
	uv run python evals/evaluate_retrieval.py

eval-batch:
	uv run python evals/run_batch.py

eval-judge:
	uv run python evals/judge.py

eval-metrics:
	uv run python evals/metrics.py evals/results_judged.json

eval-tune:
	uv run python evals/tune_agent.py

eval-logs:
	uv run python evals/logs_to_eval.py

label-app:
	uv run streamlit run evals/label_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up

ci:
	uv run python -m compileall -q .
	uv run pytest trace_replay/ -v
	uv run python evals/evaluate_retrieval.py
	uv run python evals/metrics.py evals/results_judged.json
