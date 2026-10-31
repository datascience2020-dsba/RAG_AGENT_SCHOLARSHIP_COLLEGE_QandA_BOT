import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / '.env')
load_dotenv(PROJECT.parent / '.env', override=False)
TOP_K = 5  # Maximum number of selected tools.
RETRIEVAL_K = 5  # Independent document retrieval limit.

def required(name):
    value = os.getenv(name, '').strip()
    if not value:
        raise ValueError(f'Set {name} in {PROJECT / ".env"}')
    return value

def corpus_path(name, default):
    value = os.getenv(name, '').strip()
    path = Path(value).expanduser() if value else Path(default)
    if not path.is_absolute():
        path = PROJECT.parent / path
    return path.resolve()

def state_dir():
    p = Path(os.getenv('STATE_DIR', str(PROJECT / '.agent_state')))
    if not p.is_absolute():
        p = PROJECT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
