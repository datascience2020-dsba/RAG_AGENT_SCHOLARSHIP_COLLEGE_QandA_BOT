"""Explicit paid evaluation. No fabricated success metrics or unreviewed gold labels."""
import json
from pathlib import Path
from src.agent import handle_request
from src.config import state_dir


def run(catalog, retrieval):
    cases = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))
    output = state_dir() / 'evaluation'
    output.mkdir(exist_ok=True)
    rows = []
    for case in cases:
        row = {**case, 'status': 'not_run', 'human_review': None,
               'retrieval_hit_at_5': None, 'faithfulness': None, 'abstention_correct': None}
        try:
            semantic = retrieval.search(case['question'], mode='semantic', k=5)
            hybrid = retrieval.search(case['question'], mode='hybrid', k=5)
            row['semantic_top5'] = semantic
            row['hybrid_top5'] = hybrid
            result = handle_request(case['question'], catalog, retrieval)
            row.update(status='completed_needs_review', answer=result['answer'],
                       selected_tools=result['selected_tools'], trace=result['trace'])
            gold = case.get('gold_record_ids')
            if gold:
                row['retrieval_hit_at_5'] = int(bool(set(gold) & {h['record_id'] for h in hybrid}))
            else:
                row['metric_note'] = 'No human-verified gold record IDs supplied; hit@5 remains null.'
        except Exception as exc:
            row.update(status='error', error_type=type(exc).__name__)
        rows.append(row)
        (output / 'results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Evaluation results', '', 'These are recorded runs, not automatic proof of answer quality.', '',
             '| Case | Status | Retrieval hit@5 | Review |', '|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['status']} | {r['retrieval_hit_at_5']} | Review sources, eligibility, citations and abstention |")
    lines += ['', '## How to document success/failure', '',
              'Inspect each top-five list and tool trace. Mark relevant record IDs in cases.json as gold_record_ids, then rerun.',
              'Record whether failure was source coverage, alias resolution, ranking, truncation, tool selection, stale dates, provider errors, or unsupported synthesis.',
              'Faithfulness and abstention accuracy remain null until a human or independently validated evaluator supplies judgments.']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Evaluation output:', output)
