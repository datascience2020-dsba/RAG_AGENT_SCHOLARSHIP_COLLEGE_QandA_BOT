"""Local schema adapters. Unknown schemas fail visibly rather than ingest zero rows."""
import hashlib
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import PROJECT, corpus_path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def records(path, kind):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{kind} corpus not found: {p}')
    if p.suffix.lower() == '.jsonl':
        data = [json.loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        data = json.loads(p.read_text(encoding='utf-8-sig'))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys = ('colleges', 'records', 'data') if kind == 'college' else ('scholarships', 'program_pages', 'programs', 'records', 'data')
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]
        if ('college_name' if kind == 'college' else 'program_name') in data:
            return [data]
        # Support maps keyed by identifiers if every value resembles a record.
        values = list(data.values())
        if values and all(isinstance(v, dict) and ('metadata' in v or 'program_name' in v or 'college_name' in v) for v in values):
            return values
    raise ValueError(f'Unsupported {kind} schema in {p}; expected records or a documented envelope')


class Catalog:
    def __init__(self):
        self.colleges, self.programs, self.skipped = {}, {}, []
        college_path = corpus_path(
            'COLLEGE_PATH',
            PROJECT.parent / 'stem_corpus_output' / 'data' / 'college_metadata.json',
        )
        scholarship_path = corpus_path(
            'SCHOLARSHIP_PATH',
            PROJECT.parent / 'stem_corpus_output' / 'data' / 'scholarship_metadata.json',
        )
        for position, raw in enumerate(records(college_path, 'college')):
            if not isinstance(raw, dict) or not raw.get('unitid') or not raw.get('college_name'):
                raise ValueError(f'Invalid college record at position {position}')
            row = dict(raw)
            row['unitid'] = str(row['unitid'])
            if row['unitid'] in self.colleges:
                raise ValueError(f'Duplicate college UNITID: {row["unitid"]}')
            self.colleges[row['unitid']] = row

        for position, raw in enumerate(records(scholarship_path, 'scholarship')):
            if not isinstance(raw, dict):
                raise ValueError(f'Invalid scholarship record at position {position}')
            status = raw.get('status')
            if (status is not None and status != 'accepted') or raw.get('page_type') == 'not_a_program':
                self.skipped.append({
                    'position': position,
                    'status': status if status is not None else raw.get('page_type'),
                })
                continue
            if isinstance(raw.get('metadata'), dict):
                row = {**raw['metadata']}
                for key in ('content_hash', 'raw_file'):
                    if raw.get(key):
                        row[key] = raw[key]
            else:
                row = dict(raw)
            if not row.get('program_name'):
                raise ValueError(f'Missing program_name at scholarship position {position}')
            # Page records remain distinct; matching names do not prove identical programs.
            identifier = row.get('scholarship_id') or row.get('program_id') or sha(json.dumps(row, sort_keys=True, ensure_ascii=False))[:24]
            identifier = str(identifier)
            if identifier in self.programs and self.programs[identifier] != {**row, 'program_id': identifier}:
                raise ValueError(f'Conflicting scholarship ID {identifier}')
            row['program_id'] = identifier
            row.setdefault('verification_status', 'Source metadata; live verification not performed by this app')
            self.programs[identifier] = row
        if not self.programs:
            raise ValueError('No usable scholarship records; inspect corpus schema and statuses')

    def program(self, identifier):
        if identifier not in self.programs:
            return {'error': 'Unknown program_id. Call search_scholarships first.'}
        return self.programs[identifier]

    def documents(self):
        docs = []
        for kind, records_ in [('college', self.colleges), ('scholarship', self.programs)]:
            for identifier, row in records_.items():
                title = row.get('college_name') or row.get('program_name')
                # Structural JSON text retains nulls and explicit verification caveats.
                text = f'{kind.title()}: {title}\n' + json.dumps(row, ensure_ascii=False, indent=2)
                docs.append(Document(page_content=text, metadata={
                    'record_id': identifier, 'record_type': kind,
                    'title': title, 'source_url': row.get('source_url') or row.get('website_url') or '',
                    'tier': row.get('tier') or '',
                }))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800, chunk_overlap=200,
            separators=['\n\n', '\n', '. ', ' ', ''],
        )
        chunks = []
        for doc in docs:
            for position, chunk in enumerate(splitter.split_documents([doc])):
                chunk.page_content = f'{doc.metadata["record_type"]}: {doc.metadata["title"]}\n{chunk.page_content}'
                chunk.metadata['chunk_id'] = sha(doc.metadata['record_type'] + '|' + doc.metadata['record_id'] + '|' + str(position) + '|' + chunk.page_content)
                chunks.append(chunk)
        return chunks
