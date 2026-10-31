"""Explicit CLI: prepare (local), ingest (paid), ask/chat (paid). No import-time calls."""
import argparse
import json
import os
from pathlib import Path
from src.config import state_dir
import shlex
from dotenv import load_dotenv

def load_env_file():
    env_file = Path('.env')
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        if separator:
            key = key.removeprefix('export ').strip()
            parsed = shlex.split(value, comments=True)
            os.environ.setdefault(key, parsed[0] if parsed else '')


load_env_file()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'ingest', 'ask', 'chat', 'eval'])
    parser.add_argument('--question')
    parser.add_argument('--create-index', action='store_true')
    args = parser.parse_args()
    from src.data import Catalog
    catalog = Catalog()
    if args.command == 'prepare':
        docs = catalog.documents()
        path = state_dir() / 'prepared_chunks.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for d in docs:
                f.write(json.dumps({'text': d.page_content, 'metadata': d.metadata}, ensure_ascii=False) + '\n')
        print(json.dumps({'colleges': len(catalog.colleges), 'program_pages': len(catalog.programs),
            'skipped_records': catalog.skipped, 'chunks': len(docs), 'output': str(path)}, indent=2))
        return
    from src.providers import NebiusEmbeddings
    from src.retrieval import Retrieval
    retrieval = Retrieval(catalog, NebiusEmbeddings())
    if args.command == 'ingest':
        from src.tools import TOOL_CARDS
        print(json.dumps(retrieval.ingest(TOOL_CARDS, args.create_index), indent=2))
        print('Pinecone is eventually consistent; allow a short interval before the first query. Old snapshot namespaces are retained, not deleted.')
        return
    if args.command == 'eval':
        from eval.run_eval import run
        run(catalog, retrieval)
        return
    from src.agent import handle_request
    if args.command == 'ask':
        if not args.question:
            parser.error('--question is required')
        result = handle_request(args.question, catalog, retrieval)
        print(result['answer'])
        return
    history = []
    while True:
        question = input('Student (quit to exit): ').strip()
        if question.casefold() in {'quit', 'exit'}:
            break
        try:
            result = handle_request(question, catalog, retrieval, history)
            history = result['messages']
            print('\nAssistant:', result['answer'], '\n')
        except Exception as exc:
            print(f'Request failed ({type(exc).__name__}). Check model tool-call support, schema, embedding dimension and index readiness. No automatic retry was made.')
            raise


if __name__ == '__main__':
    main()
