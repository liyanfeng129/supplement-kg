"""Offline integrity check for the committed 1,000-label reference bundle.

Run from any directory: python3 /path/to/datasets/pilot/validate_gold1000.py
Only the standard library is required. Exits nonzero on any failed check.
"""

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(directory):
    manifest = json.loads((directory / 'gold-1000.manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        require(actual == expected['sha256'], f'Checksum mismatch: {filename}')

    gold = read_rows(directory / 'gold-1000.jsonl')
    sources = read_rows(directory / 'gold-1000.sources.jsonl')
    original = read_rows(directory / 'gold-100.jsonl')
    ids = [r['dsld_id'] for r in gold]
    source_ids = [r['dsld_id'] for r in sources]
    require(len(gold) == len(set(ids)) == 1000, 'Expected exactly 1,000 unique gold rows')
    require(len(sources) == len(set(source_ids)) == 1000, 'Expected exactly 1,000 unique source rows')
    require(ids == source_ids == list(map(int, manifest['dsld_ids'])), 'Source/manifest IDs or order differ')
    require(len(original) == 100 and gold[:100] == original, 'Original gold-100 rows changed')
    require(ids == sorted(ids), 'Rows must follow the original pilot order (ascending IDs)')

    row_keys = {'dsld_id', 'idisk_id', 'name', 'purpose', 'risk',
                'purpose_confidence', 'risk_confidence', 'model', 'extracted_at'}
    audit = Counter({k: 0 for k in ('n_evidence_spans', 'n_exact_grounded',
                                  'n_statement_body_matches', 'n_header_only',
                                  'n_header_and_statement_body', 'n_ungrounded')})
    for row, source in zip(gold, sources):
        did = row['dsld_id']
        require(set(row) == row_keys, f'{did}: unexpected row schema')
        require(type(did) is int and did > 0, f'{did}: invalid DSLD ID')
        require(row['idisk_id'] is None or isinstance(row['idisk_id'], str), f'{did}: invalid iDISK ID')
        require(isinstance(row['name'], str) and row['name'] == source['name'], f'{did}: name differs from source')
        require(isinstance(row['model'], str) and row['model'].strip(), f'{did}: missing model provenance')
        stamp = datetime.fromisoformat(row['extracted_at'].replace('Z', '+00:00'))
        require(stamp.tzinfo is not None, f'{did}: extraction timestamp needs a timezone')
        passage = source['passage']
        header, _, body = passage.partition('\nLABEL STATEMENTS:')
        body = re.sub(r'^\[[^\]]+\] ', '', body, flags=re.M)
        for field in ('purpose', 'risk'):
            entries, confidence = row[field], row[field + '_confidence']
            require(isinstance(entries, list), f'{did}: {field} must be a list')
            require(type(confidence) is int and 0 <= confidence <= 5, f'{did}: invalid confidence')
            require(bool(entries) == bool(confidence), f'{did}: entries/confidence disagree')
            for entry in entries:
                require(set(entry) == {'text', 'evidence', 'grounded'}, f'{did}: invalid entry schema')
                require(isinstance(entry['text'], str) and entry['text'].strip(), f'{did}: blank claim')
                evidence = entry['evidence']
                require(isinstance(evidence, str) and evidence.strip(), f'{did}: blank evidence')
                require(evidence in passage, f'{did}: evidence is not an exact source substring: {evidence!r}')
                require(entry['grounded'] is True, f'{did}: grounded flag must be true')
                audit['n_evidence_spans'] += 1
                audit['n_exact_grounded'] += evidence in passage
                audit['n_statement_body_matches'] += evidence in body
                audit['n_header_only'] += evidence in header and evidence not in body
                audit['n_header_and_statement_body'] += evidence in header and evidence in body
                audit['n_ungrounded'] += evidence not in passage

    counts = {
        'n_requested': 1000, 'n_extracted': len(gold), 'n_unique_ids': len(set(ids)),
        'n_with_purpose': sum(bool(r['purpose']) for r in gold),
        'n_with_risk': sum(bool(r['risk']) for r in gold),
        'n_purpose_entries': sum(len(r['purpose']) for r in gold),
        'n_risk_entries': sum(len(r['risk']) for r in gold),
        'n_ungrounded_entries': audit['n_ungrounded'],
    }
    for key, value in counts.items():
        require(manifest[key] == value, f'Manifest count differs: {key}')
    require(dict(Counter(r['model'] for r in gold)) == manifest['model_counts'], 'Model counts differ')
    require(not manifest['errors'], 'Manifest contains errors')
    for key, value in audit.items():
        require(manifest['header_audit'][key] == value, f'Header audit differs: {key}')
    require(audit['n_statement_body_matches'] == audit['n_evidence_spans'], 'Evidence requires headers or type prefixes')

    provenance = manifest['provenance']
    codex_ids = provenance['codex_dsld_ids']
    recovered_ids = provenance['recovered_claude_dsld_ids']
    require(len(codex_ids) == len(set(codex_ids)) == provenance['new_codex_rows'] == 15,
            'Codex cohort count differs')
    require(len(recovered_ids) == len(set(recovered_ids)) == provenance['recovered_claude_draft_rows'] == 75,
            'Recovered cohort count differs')
    require(not set(codex_ids) & set(recovered_ids), 'Recovery cohorts overlap')
    by_id = {r['dsld_id']: r for r in gold}
    require(set(codex_ids) == {r['dsld_id'] for r in gold if r['model'] == 'codex (interactive)'},
            'Codex model attribution differs')
    require(set(recovered_ids) <= set(ids[100:]), 'Recovered ID outside expansion')
    for edit in provenance['review_edits']:
        actual = by_id[edit['dsld_id']][edit['field']]
        if isinstance(actual, list):
            actual = [{k: v for k, v in e.items() if k != 'grounded'} for e in actual]
        require(actual == edit['after'], f'Review log differs for {edit["dsld_id"]}')
    return {**counts, 'original_100_preserved': True,
            'all_checksums_match': True, 'evidence_spans_checked': audit['n_evidence_spans'],
            'header_only_evidence_spans': audit['n_header_only']}


if __name__ == '__main__':
    print(json.dumps(validate(Path(__file__).resolve().parent), indent=2))
