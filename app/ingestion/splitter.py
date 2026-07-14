from __future__ import annotations 
 
import re 
 
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
 
_HEADING_RE = re.compile(r'(#{1,6})\s+(.*)$') 
 
_INTENT_RULES = [ 
    ('red_flag', ('红旗', '急诊', '不要等', '立即就医', '优先急诊')), 
    ('prep', ('检查准备', '就诊准备', '建议带', '带上', '检查当天')), 
    ('followup', ('复诊', '生活管理', '长期观察', '什么时候不要等复诊')), 
    ('symptom_log', ('症状记录', '就医沟通', '时间线', '记录什么')), 
    ('faq', ('常见问题', '误区')), 
    ('route', ('建议挂号科室', '分诊指南', '导诊', '主候选', '备选路径')), 
] 
 
def build_text_splitter(chunk_size: int = 320, chunk_overlap: int = 60): 
    return RecursiveCharacterTextSplitter( 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=['\n\n', '\n', '。', '；', '！', '？', '，', ' ', ''], 
    ) 
 
def _iter_sections(text: str, default_title: str): 
    sections = [] 
    current_title = default_title 
    current_lines = [] 
    for raw_line in text.splitlines(): 
        line = raw_line.rstrip() 
        match = _HEADING_RE.match(line.strip()) 
        if match: 
            if current_lines: 
                body = '\n'.join(current_lines).strip() 
                if body: 
                    sections.append((current_title, body)) 
            current_title = match.group(2).strip() or default_title 
            current_lines = [] 
            continue 
        current_lines.append(line) 
    if current_lines: 
        body = '\n'.join(current_lines).strip() 
        if body: 
            sections.append((current_title, body)) 
    return sections or [(default_title, text.strip())] 
 
def _infer_intent_type(file_stem: str, title: str, content: str) -> str: 
    text = ' '.join([file_stem or '', title or '', content[:120]]) 
    for intent, keywords in _INTENT_RULES: 
        if any(keyword in text for keyword in keywords): 
            return intent 
    return 'general' 
 
def split_documents(documents, chunk_size: int = 320, chunk_overlap: int = 60): 
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    chunks = [] 
    for doc in documents: 
        metadata = dict(doc.metadata) 
        default_title = str(metadata.get('section', metadata.get('file_stem', ''))).strip() 
        file_stem = str(metadata.get('file_stem', '')).strip() 
        doc_id = str(metadata.get('doc_id', file_stem)).strip() 
        sections = _iter_sections(doc.page_content.strip(), default_title) 
        for section_index, (section_title, section_body) in enumerate(sections): 
            parent_id = f'{doc_id}::section::{section_index}' 
            intent_type = _infer_intent_type(file_stem=file_stem, title=section_title, content=section_body) 
            section_chunks = [part.strip() for part in splitter.split_text(section_body) if part.strip()] 
            if not section_chunks: 
                section_chunks = [section_body.strip()] 
            total = len(section_chunks) 
            for idx, piece in enumerate(section_chunks): 
                content = piece if piece.startswith(section_title) else f'{section_title}\n\n{piece}' 
                item_meta = dict(metadata) 
                item_meta['section_title'] = section_title 
                item_meta['parent_id'] = parent_id 
                item_meta['parent_title'] = section_title 
                item_meta['intent_type'] = intent_type 
                item_meta['chunk_id'] = idx 
                item_meta['chunk_count'] = total 
                item_meta['chunk_part'] = idx 
                chunks.append(Document(page_content=content, metadata=item_meta)) 
    return chunks 
