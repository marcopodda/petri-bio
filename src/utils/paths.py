import pyrootutils

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
PROJECT_ROOT = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
DATA_DIR = PROJECT_ROOT / "data"
EXPS_DIR = PROJECT_ROOT / "experiments"
CONFIG_DIR = PROJECT_ROOT / "config"

PATHWATYS_DIR = DATA_DIR / "pathways"
PATHWAY_PATHS = sorted(PATHWATYS_DIR.glob("*.dot"))
