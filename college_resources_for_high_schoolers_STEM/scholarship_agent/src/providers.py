"""Nebius-only embeddings through LangChain's Embeddings interface."""
import json
import math
import sqlite3
from urllib.parse import urlsplit
import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from .config import required, state_dir
from .data import sha


def base_url():
    value = required('NEBIUS_BASE_URL').rstrip('/')
    p = urlsplit(value)
    if p.scheme != 'https' or not p.hostname or not (p.hostname == 'nebius.com' or p.hostname.endswith('.nebius.com')) or p.username or p.password or p.query or p.fragment or p.path != '/v1':
        raise ValueError('NEBIUS_BASE_URL must be a plain HTTPS Nebius URL ending /v1; remove Markdown formatting')
    return value


class NebiusEmbeddings(Embeddings):
    def __init__(self):
        self.model = required('NEBIUS_EMBED_MODEL')
        self.base = base_url()
        self.key = required('NEBIUS_API_KEY')
        self.db = sqlite3.connect(state_dir() / 'embeddings.sqlite3', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors(k TEXT PRIMARY KEY, vector TEXT)')
        self.dimension = None

    def embed_documents(self, texts):
        output = [None] * len(texts)
        missing = []
        for index, text in enumerate(texts):
            key = sha(self.base + '|' + self.model + '|' + text)
            row = self.db.execute('SELECT vector FROM vectors WHERE k=?', (key,)).fetchone()
            if row:
                output[index] = json.loads(row[0])
            else:
                missing.append((index, key, text))
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            response = requests.post(self.base + '/embeddings', headers={'Authorization': 'Bearer ' + self.key},
                                     json={'model': self.model, 'input': [x[2] for x in batch], 'encoding_format': 'float'}, timeout=(10, 90))
            response.raise_for_status()
            data = sorted(response.json()['data'], key=lambda x: x['index'])
            if len(data) != len(batch) or [x['index'] for x in data] != list(range(len(batch))):
                raise ValueError('Embedding response indices do not match the input batch')
            for (index, key, _), item in zip(batch, data):
                vector = item['embedding']
                if not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                    raise ValueError('Invalid embedding values')
                output[index] = vector
                self.db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?)', (key, json.dumps(vector)))
            self.db.commit()
        dimensions = {len(v) for v in output}
        if len(dimensions) > 1:
            raise ValueError('Inconsistent embedding dimensions')
        if output:
            dimension = len(output[0])
            if self.dimension is not None and self.dimension != dimension:
                raise ValueError('Embedding model dimension changed')
            self.dimension = dimension
        return output

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def chat_model():
    # OpenAI-compatible CLIENT; every request goes to the configured Nebius endpoint.
    return ChatOpenAI(model=required('NEBIUS_LLM_MODEL'), api_key=required('NEBIUS_API_KEY'),
                      base_url=base_url(), temperature=0, max_retries=0, timeout=90,
                      use_responses_api=False)
