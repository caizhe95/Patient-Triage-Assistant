from __future__ import annotations 
 
import json 
import pickle 
 
import jieba 
import numpy as np 
from qdrant_client.http import models as qmodels 
from rank_bm25 import BM25Okapi 
 
from app.config import get_settings 
from app.embeddings import get_embedder, normalize 
from app.ingestion.knowledge_layout import SHARED_DOMAIN, build_shared_and_domain_documents 
from app.ingestion.loader import load_raw_documents 
from app.ingestion.splitter import split_documents 
from app.retrieval.qdrant_utils import create_qdrant_client 
from app.utils.logger import get_logger 
 
logger = get_logger(__name__) 
EXCLUDED_INDEX_DOMAINS = {'emergency'}
 
def _domain_names(settings): 
    raw_dir = settings.raw_data_dir 
    if not raw_dir.exists(): 
        return [] 
    return sorted([path.name for path in raw_dir.iterdir() if path.is_dir() and path.name not in EXCLUDED_INDEX_DOMAINS]) 
 
def _index_paths(settings, domain): 
    return { 
        'docs': settings.qdrant_storage_dir / f'{domain}_docs.json', 
        'encoder': settings.qdrant_storage_dir / f'{domain}_encoder.json', 
        'bm25': settings.bm25_index_dir / f'{domain}.pkl', 
    } 
 
def _collection_name(settings, domain): 
    return f'{settings.qdrant_collection_prefix}_{domain}' 
 
def _serialize_documents(documents): 
    return [{'content': doc.page_content, 'metadata': dict(doc.metadata)} for doc in documents] 
 
def _tokenize(text): 
    return jieba.lcut(text) 
 
def _write_json(path, payload): 
    path.parent.mkdir(parents=True, exist_ok=True) 
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
 
def _write_bm25(path, documents): 
    path.parent.mkdir(parents=True, exist_ok=True) 
    serialized = _serialize_documents(documents) 
    tokenized_corpus = [_tokenize(doc['content']) for doc in serialized] 
    bm25 = BM25Okapi(tokenized_corpus) 
    with path.open('wb') as fh: 
        pickle.dump({'documents': serialized, 'tokenized_corpus': tokenized_corpus, 'bm25': bm25}, fh) 
 
def _write_qdrant_collection(client, settings, domain, documents, vectors, batch_size=16): 
    collection_name = _collection_name(settings, domain) 
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=int(vectors.shape[1]), distance=qmodels.Distance.COSINE),
        timeout=60,
    ) 
    points = [] 
    for idx, (doc, vector) in enumerate(zip(documents, vectors)): 
        points.append(qmodels.PointStruct(id=idx, vector=vector.tolist(), payload={'content': doc.page_content, 'metadata': dict(doc.metadata)})) 
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
 
def _write_index_bundle(client, settings, domain, documents, embedder): 
    paths = _index_paths(settings, domain) 
    serialized = _serialize_documents(documents) 
    _write_json(paths['docs'], serialized) 
    _write_json(paths['encoder'], {'backend': getattr(embedder, 'backend', 'tfidf'), 'model_name': getattr(embedder, 'model_name', 'tfidf-jieba-bigram'), 'collection_name': _collection_name(settings, domain), 'distance': 'cosine'}) 
    _write_bm25(paths['bm25'], documents) 
    vectors = normalize(embedder.encode_documents([doc['content'] for doc in serialized]).astype(np.float32)) 
    _write_qdrant_collection(client, settings, domain, documents, vectors)
 
def indexes_exist(): 
    settings = get_settings() 
    domains = [SHARED_DOMAIN] + _domain_names(settings) 
    for domain in domains: 
        paths = _index_paths(settings, domain) 
        if not paths['docs'].exists() or not paths['encoder'].exists() or not paths['bm25'].exists(): 
            return False 
    return True 
 
def build_all_indexes(): 
    settings = get_settings() 
    settings.qdrant_storage_dir.mkdir(parents=True, exist_ok=True) 
    settings.bm25_index_dir.mkdir(parents=True, exist_ok=True) 
 
    raw_documents = [
        doc for doc in load_raw_documents(settings.raw_data_dir)
        if str(doc.metadata.get('domain', '')) not in EXCLUDED_INDEX_DOMAINS
    ]
    if not raw_documents: 
        raise ValueError(f'No raw documents found in {settings.raw_data_dir}') 
 
    chunked_documents = split_documents(raw_documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap) 
    shared_docs, domain_docs = build_shared_and_domain_documents(chunked_documents) 
 
    groups = [(SHARED_DOMAIN, shared_docs)] 
    groups.extend((domain, domain_docs[domain]) for domain in sorted(domain_docs)) 
 
    all_documents = [] 
    for _, documents in groups: 
        all_documents.extend(documents) 
 
    if not all_documents: 
        raise ValueError('No documents produced after chunking') 
 
    embedder = get_embedder() 
    _ = embedder.encode_documents([doc.page_content for doc in all_documents]) 
    client = create_qdrant_client(settings) 
 
    for domain, documents in groups: 
        if not documents: 
            continue 
        logger.info('Indexing %s (%s docs)', domain, len(documents)) 
        _write_index_bundle(client, settings, domain, documents, embedder)
