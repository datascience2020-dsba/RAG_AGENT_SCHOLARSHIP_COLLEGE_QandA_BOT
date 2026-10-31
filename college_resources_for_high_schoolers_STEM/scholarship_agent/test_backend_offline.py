"""No provider imports or network: test the actual adapter methods with fixtures."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('notebook_backend.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)

def definition(name, namespace):
    node = next(n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'backend_fixture', 'exec'), namespace)
    return namespace[name]

class OfflineTests(unittest.TestCase):
    def test_files_have_valid_python_syntax(self):
        for name in ['notebook_backend.py', 'streamlit_app.py', 'diagnose_retrieval.py']:
            ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))

    def test_retrieval_wrapper_remains_read_only(self):
        cls = definition('ReadOnlyIndex', {'NAMESPACE_MAP': {'scholarships': 'existing-scholarships', 'colleges': 'existing-colleges'}})
        self.assertFalse(hasattr(cls(object()), 'upsert'))

    def test_ingestion_requires_explicit_confirmation(self):
        called = []
        ingest = definition('ingest_corpus', {
            'connect_pinecone': lambda: called.append('connected'),
        })
        with self.assertRaisesRegex(ValueError, 'confirm=True'):
            ingest()
        self.assertEqual(called, [])

    def test_ingestion_refreshes_changed_and_adds_missing_chunks(self):
        class FakeIndex:
            def __init__(self):
                self.vectors = {
                    'unchanged': {'metadata': {'page_content': 'same text'}},
                    'changed': {'metadata': {'page_content': 'old text'}},
                }
                self.upserted = []

            def fetch(self, ids, namespace):
                return SimpleNamespace(vectors={
                    chunk_id: self.vectors[chunk_id]
                    for chunk_id in ids if chunk_id in self.vectors
                })

            def upsert(self, vectors, namespace):
                self.upserted.extend(vectors)

        index = FakeIndex()
        docs = [
            SimpleNamespace(metadata={'chunk_id': 'unchanged'}, page_content='same text'),
            SimpleNamespace(metadata={'chunk_id': 'changed'}, page_content='new text'),
            SimpleNamespace(metadata={'chunk_id': 'missing'}, page_content='added text'),
        ]

        class Embeddings:
            def embed_documents(self, texts):
                return [[float(len(text))] for text in texts]

        ingest = definition('ingest_corpus', {
            'connect_pinecone': lambda: None,
            'PINECONE_RAW_INDEX': index,
            'NAMESPACES': ('scholarships',),
            'NAMESPACE_MAP': {'scholarships': 'physical-scholarships'},
            'chunks_by_namespace': {'scholarships': docs},
            'field': lambda obj, name, default=None: obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default),
            'get_embeddings': Embeddings,
            'safe_metadata_value': lambda value: value,
        })
        result = ingest(confirm=True)
        self.assertEqual(result[0]['upserted'], 2)
        self.assertEqual(result[0]['updated_existing'], 1)
        self.assertEqual(result[0]['skipped_unchanged'], 1)
        self.assertEqual({row['id'] for row in index.upserted}, {'changed', 'missing'})

    def test_physical_namespace_mapping(self):
        class Index:
            def query(self, **kwargs): return kwargs
        cls = definition('ReadOnlyIndex', {'NAMESPACE_MAP': {'scholarships': 'existing-scholarships', 'colleges': 'existing-colleges'}})
        idx = cls(Index())
        self.assertEqual(idx.query(namespace='scholarships')['namespace'], 'existing-scholarships')
        self.assertFalse(hasattr(idx, 'upsert'))

    def fixture(self, changed=False, missing_text=False):
        document = SimpleNamespace(page_content='Edison Scholars award and evidence', metadata={'chunk_id': 'same-id', 'program_name': 'Edison Scholars'})
        class BM25:
            def get_scores(self, terms): return [3.0]
        class Embed:
            def embed_query(self, query): return [.1, .2]
        class Index:
            def query(self, **kwargs):
                text = 'changed text' if changed else document.page_content
                metadata = {'program_name': 'Edison Scholars'}
                if not missing_text: metadata['page_content'] = text
                return SimpleNamespace(matches=[SimpleNamespace(id='same-id', metadata=metadata, score=.8)])
        env = {'TOP_K': 5, 'NAMESPACES': ('scholarships', 'colleges'), 'DRY_RUN': False,
               'RRF_CONSTANT': 60, 'connect_pinecone': lambda: Index(), 'get_embeddings': lambda: Embed(),
               'native_pinecone_filter': lambda f, n: None, 'metadata_matches': lambda *a: True,
               'bm25_documents': {'scholarships': [document]}, 'bm25_by_namespace': {'scholarships': BM25()},
               'tokens': lambda x: x.split()}
        return definition('retrieve', env)

    def test_hybrid_fuses_dense_and_keyword_same_id(self):
        result = self.fixture()('Edison', 'scholarships')
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['score'], 2 / 61)
        self.assertEqual(result[0]['dense_score'], .8)
        self.assertEqual(result[0]['bm25_score'], 3.0)

    def test_reject_local_remote_text_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.fixture(changed=True)('Edison', 'scholarships')

    def test_reject_wrong_metadata_schema(self):
        with self.assertRaisesRegex(ValueError, 'page_content'):
            self.fixture(missing_text=True)('Edison', 'scholarships')

if __name__ == '__main__':
    unittest.main()
