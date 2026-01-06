#!/usr/bin/env bash

set -e

export PYTHONUNBUFFERED=1

for cfg in configs/your_config_file_gene_*.yaml; do
	name=$(basename "$cfg" .yaml)
	echo "Running $cfg"

	python3 src/sk_tune.py --config "$cfg" \

	echo "Finished $cfg"

done

echo "All runs completed"
