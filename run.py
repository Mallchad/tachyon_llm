
import source.tachyon_library

execute_list( [
    "python3",
    "-m",
    "mypy",
    "--incremental",
    "source/experiments.py"
])
# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
execute_list( [
    "python3",
    "source/experiments.py"
])
