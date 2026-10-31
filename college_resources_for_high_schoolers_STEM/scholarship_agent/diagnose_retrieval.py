"""Read-only diagnostic. Running this uses query embeddings and Pinecone reads, no LLM or ingestion."""
from notebook_backend import connection_details, retrieve
import json

if __name__ == '__main__':
    print(json.dumps(connection_details(), indent=2))
    # Separate queries test both named programs instead of allowing one to dominate.
    for name in ['Edison Scholars', 'SMART Scholarship-for-Service']:
        print('\nPROGRAM:', name)
        hits = retrieve(name, 'scholarships', k=5)
        for hit in hits:
            print(json.dumps({'chunk_id': hit['chunk_id'], 'rrf': hit['score'],
                              'dense_score': hit['dense_score'], 'bm25_score': hit['bm25_score'],
                              'program_name': hit['metadata'].get('program_name'),
                              'source_url': hit['metadata'].get('source_url'),
                              'preview': hit['page_content'][:300]}, indent=2))
