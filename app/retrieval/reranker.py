  
from __future__ import annotations  
 
from abc import ABC, abstractmethod  
 
from app.config import get_settings  
from app.utils.logger import get_logger  
 
logger = get_logger(__name__)  
 
 
def _doc_key(doc):  
    metadata = doc.get('metadata', {}) or {}  
    if metadata.get('parent_id'):  
        return str(metadata.get('parent_id'))  
    if metadata.get('doc_id'):  
        return str(metadata.get('doc_id'))  
    return '{}::{}'.format(metadata.get('domain', ''), metadata.get('chunk_id', ''))  
 
 
def _dedup_docs(docs, top_k):  
    seen = set()  
    out = []  
    for doc in docs:  
        key = _doc_key(doc)  
        if key in seen:  
            continue  
        seen.add(key)  
        out.append(doc)  
        if len(out) >= top_k:  
            break  
    return out  
 
 
class BaseReranker(ABC):  
    @abstractmethod  
    def rerank(self, query, docs, top_k=5):  
        raise NotImplementedError  
 
 
class DefaultReranker(BaseReranker):  
    def rerank(self, query, docs, top_k=5):  
        ranked = sorted(docs, key=lambda item: float(item.get('fused_score', item.get('score', 0.0))), reverse=True)  
        return _dedup_docs(ranked, top_k)  
 
 
class BGEReranker(BaseReranker):  
    def __init__(self, model_name=None):  
        settings = get_settings()  
        self.model_name = model_name or settings.rerank_model  
        self.model = None  
        try:  
            from sentence_transformers import CrossEncoder  
            self.model = CrossEncoder(self.model_name)  
        except Exception as exc:  
            logger.warning('failed to load bge reranker %s, fallback to default: %s', self.model_name, exc)  
            self.model = None  
 
    def available(self):  
        return self.model is not None  
 
    def rerank(self, query, docs, top_k=5):  
        if not docs:  
            return []  
        if self.model is None:  
            return DefaultReranker().rerank(query, docs, top_k=top_k)  
        scores = self.model.predict([(query, str(doc.get('content', ''))) for doc in docs])  
        reranked = []  
        for doc, score in zip(docs, scores):  
            item = dict(doc)  
            item['rerank_score'] = float(score)  
            reranked.append(item)  
        reranked.sort(key=lambda item: float(item.get('rerank_score', 0.0)), reverse=True)  
        return _dedup_docs(reranked, top_k)  
