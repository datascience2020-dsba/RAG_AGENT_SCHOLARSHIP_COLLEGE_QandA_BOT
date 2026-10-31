import json
import os
import re
import time
from collections import defaultdict
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from .config import required, state_dir, RETRIEVAL_K
from .data import sha


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.casefold())


class Retrieval:
    def __init__(self, catalog, embeddings):
        self.catalog, self.embeddings = catalog, embeddings
        self.chunks = catalog.documents()
        if not self.chunks:
            raise ValueError('No documents to index')
        self.by_id = {d.metadata['chunk_id']: d for d in self.chunks}
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.chunks])
        # Content-addressed namespaces prevent stale versions mixing with current data.
        self.snapshot = sha('|'.join(sorted(self.by_id)) + '|' + embeddings.model)[:20]
        self.doc_namespace = 'docs-' + self.snapshot
        self.tool_namespace = 'tools-' + self.snapshot
        self.pc = None
        self.index = None

    def connect(self):
        if self.index is not None:
            return
        self.pc = Pinecone(api_key=required('PINECONE_API_KEY'))
        name = required('PINECONE_INDEX')
        info = self.pc.describe_index(name)
        dimension = len(self.embeddings.embed_query('dimension compatibility check'))
        if int(info.dimension) != dimension or info.metric != 'cosine':
            raise ValueError('Pinecone must be a cosine vector index matching the configured Nebius embedding dimension')
        self.index = self.pc.Index(host=info.host)

    def ingest(self, tool_cards, create_index=False):
        dimension = len(self.embeddings.embed_query('dimension compatibility check'))
        pc = Pinecone(api_key=required('PINECONE_API_KEY'))
        name = required('PINECONE_INDEX')
        names = pc.list_indexes().names()
        if name not in names:
            if not create_index:
                raise ValueError('Index does not exist. Pass --create-index to create a dedicated vector index')
            pc.create_index(name=name, dimension=dimension, metric='cosine',
                            spec=ServerlessSpec(cloud=os.getenv('PINECONE_CLOUD', 'aws'), region=os.getenv('PINECONE_REGION', 'us-east-1')))
        for _ in range(60):
            if pc.describe_index(name).status['ready']:
                break
            time.sleep(2)
        else:
            raise TimeoutError('Pinecone index was not ready within 120 seconds')
        self.connect()
        for start in range(0, len(self.chunks), 32):
            batch = self.chunks[start:start + 32]
            vectors = self.embeddings.embed_documents([d.page_content for d in batch])
            self.index.upsert(vectors=[{'id': d.metadata['chunk_id'], 'values': v,
                'metadata': {**d.metadata, 'text': d.page_content}} for d, v in zip(batch, vectors)], namespace=self.doc_namespace)
        vectors = self.embeddings.embed_documents([c['description'] for c in tool_cards])
        self.index.upsert(vectors=[{'id': c['name'], 'values': v, 'metadata': c} for c, v in zip(tool_cards, vectors)], namespace=self.tool_namespace)
        manifest = {'snapshot': self.snapshot, 'embedding_model': self.embeddings.model,
                    'dimension': dimension, 'documents': len(self.chunks), 'index': name,
                    'doc_namespace': self.doc_namespace, 'tool_namespace': self.tool_namespace}
        (state_dir() / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest

    def require_manifest(self):
        path = state_dir() / 'manifest.json'
        if not path.exists():
            raise ValueError(
                f'Pinecone ingestion is required for semantic/hybrid retrieval, but no manifest exists at {path}. '
                'From the scholarship_agent directory, run "python app.py ingest" using your configured Pinecone index. '
                'Ingestion calls the embedding API and writes vectors to Pinecone.'
            )
        m = json.loads(path.read_text(encoding='utf-8'))
        if m['snapshot'] != self.snapshot or m['index'] != required('PINECONE_INDEX'):
            raise ValueError(
                'The local corpus, embedding model, or Pinecone index differs from the ingested snapshot. '
                'From the scholarship_agent directory, run "python app.py ingest" to index the current snapshot.'
            )

    def search(self, query, record_type='scholarship', mode='hybrid', k=RETRIEVAL_K):
        if record_type not in {'scholarship', 'college'} or mode not in {'hybrid', 'semantic', 'keyword'}:
            raise ValueError('Invalid record_type or retrieval mode')
        k = max(1, min(int(k), 20))
        rankings = []
        if mode != 'keyword':
            self.require_manifest(); self.connect()
            response = self.index.query(namespace=self.doc_namespace,
                vector=self.embeddings.embed_query(query), top_k=max(20, k),
                include_metadata=True, filter={'record_type': {'$eq': record_type}})
            rankings.append([m.id for m in response.matches if m.id in self.by_id])
        if mode != 'semantic':
            scores = self.bm25.get_scores(tokenize(query))
            rankings.append([self.chunks[i].metadata['chunk_id'] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                             if scores[i] > 0 and self.chunks[i].metadata['record_type'] == record_type][:max(20, k)])
        fused = defaultdict(float)
        for ranking in rankings:
            for rank, identifier in enumerate(ranking, 1):
                fused[identifier] += 1 / (60 + rank)
        result = []
        for identifier in sorted(fused, key=fused.get, reverse=True)[:k]:
            doc = self.by_id[identifier]
            result.append({'citation_id': identifier, 'score': fused[identifier],
                           **doc.metadata, 'text': doc.page_content})
        return result
