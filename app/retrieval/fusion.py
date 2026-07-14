  
from __future__ import annotations  
 
from collections import defaultdict  
 
 
def _doc_key(doc):  
    metadata = doc.get('metadata', {}) or {}  
    parent_id = metadata.get('parent_id')  
    if parent_id:  
        return f"{metadata.get('domain', '')}::{parent_id}"  
    return f"{metadata.get('domain', '')}::{metadata.get('doc_id', '')}::{metadata.get('chunk_id', '')}"  
 
 
def rrf_fuse(ranked_lists, k=60, limit=10):  
    fused_scores = defaultdict(float)  
    doc_bank = {}  
    source_bank = defaultdict(set)  
    for source_name, docs in ranked_lists.items():  
        for rank, doc in enumerate(docs, start=1):  
            key = _doc_key(doc)  
            doc_bank[key] = doc_bank.get(key, doc)  
            fused_scores[key] += 1.0 / (k + rank)  
            source_bank[key].add(source_name)  
    fused = []  
    for key, score in fused_scores.items():  
        doc = dict(doc_bank[key])  
        doc['fused_score'] = score  
        doc['retrieval_source'] = 'fused'  
        metadata = dict(doc.get('metadata', {}))  
        metadata['retrieval_sources'] = sorted(source_bank[key])  
        doc['metadata'] = metadata  
        fused.append(doc)  
    fused.sort(key=lambda item: float(item.get('fused_score', 0.0)), reverse=True)  
    return fused[:limit]  
