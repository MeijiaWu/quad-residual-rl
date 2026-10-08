.PHONY: test gains eval sysid bench train seeds aggregate experiments report parity supervisor isaac-smoke isaac-suite isaac-report clean

SEEDS ?= 0 1 2 3 4 5 6 7 8 9
MODE  ?= residual

test:
	python3 tests/test_core.py

gains:
	python3 scripts/select_baseline_gains.py

eval:
	python3 scripts/eval.py

sysid:
	python3 scripts/sysid.py

bench:
	python3 scripts/bench_latency.py

train:
	python3 scripts/train_sim_lite.py --mode $(MODE) --seed 0 --out results/$(MODE)_s0

# Train and evaluate one run per seed, then aggregate (make seeds MODE=direct)
seeds:
	for s in $(SEEDS); do \
	  python3 scripts/train_sim_lite.py --mode $(MODE) --seed $$s --out results/$(MODE)_s$$s && \
	  python3 scripts/eval.py --policy results/$(MODE)_s$$s.zip --out results/eval_$(MODE)_s$$s.json ; \
	done
	python3 scripts/aggregate.py results/eval_$(MODE)_s*.json

aggregate:
	python3 scripts/aggregate.py results/eval_$(MODE)_s*.json

# Everything: main comparison, ablations, generalization tests, then results/report.md
JOBS ?= 1
experiments:
	python3 scripts/experiments.py run main ablations generalization stress supervisor --seeds 0-9 --jobs $(JOBS) --report

# OOD supervisor: lag-estimator calibration (CPU, seconds)
supervisor:
	python3 scripts/check_supervisor.py

report:
	python3 scripts/experiments.py report

# Isaac path: CPU parity of the torch core (no Isaac needed), then an Isaac smoke test
parity:
	python3 scripts/check_torch_core.py --policy results/direct_s0.zip

isaac-smoke:
	python3 scripts/train_isaac.py --mode direct --num_envs 64 --max_iterations 5 --headless

# Isaac follow-ups on the 10+10 trained runs: OOD supervisor + learning curve, then the report
isaac-suite:
	python3 scripts/isaac_suite.py run ood curve --jobs $(JOBS)
	python3 scripts/isaac_suite.py report

isaac-report:
	python3 scripts/isaac_suite.py report

# Removes evaluation outputs only; trained checkpoints are kept.
clean:
	rm -rf results/*.json results/eval results/report.md __pycache__ */__pycache__
	find . -name '*.pyc' -delete
