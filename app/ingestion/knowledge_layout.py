  
from collections import defaultdict  
 
from langchain_core.documents import Document  
 
SHARED_DOMAIN = 'shared'  
 
SHARED_TEMPLATE_SUFFIXES = {  
    '常见问题与误区': 'common_faq',  
    '检查结果与就医路径': 'result_path',  
    '生活管理与复诊建议': 'followup_care',  
    '症状记录与就医沟通': 'symptom_log',  
    '检查准备': 'prep',  
}  
 
GENERIC_TEMPLATE_LINES = {  
    '尽量把饮食、睡眠、用药和症状记录整理成一页，方便医生快速判断。',  
    '复诊不是简单重复挂号，而是看趋势、看疗效、看是否需要换科或换检查。',  
    '症状记录不是为了替代医生，而是为了让医生更快抓住重点。',  
    '若出现生命体征不稳、持续加重或功能快速受损，优先急诊。',  
    '一旦在等待检查期间症状明显升级，应直接转急诊。',  
}  
 
 
def shared_family_for_stem(stem):  
    for suffix, family in SHARED_TEMPLATE_SUFFIXES.items():  
        if stem.endswith(suffix):  
            return family  
    return None  
 
 
def _shared_doc_priority(doc):  
    metadata = doc.metadata  
    domain = str(metadata.get('domain', ''))  
    return (1 if domain == 'general' else 0, len(str(doc.page_content)), str(metadata.get('source', '')))  
 
 
def _normalize_line(line):  
    text = str(line or '').strip()  
    if text.startswith('- '):  
        text = text[2:].strip()  
    return text  
 
 
def _strip_generic_templates(content):  
    cleaned_lines = []  
    removed = []  
    for line in str(content or '').splitlines():  
        normalized = _normalize_line(line)  
        if normalized in GENERIC_TEMPLATE_LINES:  
            removed.append(normalized)  
            continue  
        cleaned_lines.append(line)  
    cleaned = chr(10).join(cleaned_lines).strip()  
    return cleaned, removed  
 
 
def build_shared_and_domain_documents(documents):  
    shared_buckets = defaultdict(list)  
    domain_docs = defaultdict(list)  
    for doc in documents:  
        metadata = dict(doc.metadata)  
        stem = str(metadata.get('file_stem', metadata.get('doc_id', '')))  
        family = shared_family_for_stem(stem)  
        if family:  
            shared_buckets[family].append(doc)  
            continue  
        domain = str(metadata.get('domain', 'general'))  
        cleaned_content, removed = _strip_generic_templates(doc.page_content)  
        if not cleaned_content:  
            cleaned_content = doc.page_content  
        metadata['knowledge_layer'] = 'domain'  
        metadata['file_stem'] = stem  
        if removed:  
            metadata['template_deduped'] = removed  
        domain_docs[domain].append(Document(page_content=cleaned_content, metadata=metadata))  
    shared_docs = []  
    for family in sorted(shared_buckets):  
        preferred = max(shared_buckets[family], key=_shared_doc_priority)  
        metadata = dict(preferred.metadata)  
        metadata['domain'] = SHARED_DOMAIN  
        metadata['source_domain'] = preferred.metadata.get('domain', '')  
        metadata['knowledge_layer'] = 'shared'  
        metadata['shared_family'] = family  
        metadata['file_stem'] = str(metadata.get('file_stem', metadata.get('doc_id', '')))  
        metadata['doc_id'] = f'shared:{family}'  
        shared_docs.append(Document(page_content=preferred.page_content, metadata=metadata))  
    return shared_docs, domain_docs  
