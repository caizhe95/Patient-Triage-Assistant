from pathlib import Path 
 
from langchain_core.documents import Document 
 
from app.ingestion.knowledge_layout import shared_family_for_stem 
 
SUPPORTED_SUFFIXES = {'.md', '.txt'} 
 
def load_raw_documents(raw_dir): 
    documents = [] 
    for domain_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]): 
        for file_path in sorted(domain_dir.rglob('*')): 
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES: 
                continue 
            content = file_path.read_text(encoding='utf-8', errors='ignore').strip() 
            if not content: 
                continue 
            section = file_path.stem 
            for line in content.splitlines(): 
                if line.startswith('#'): 
                    section = line.lstrip('#').strip() 
                    break 
            file_stem = file_path.stem 
            shared_family = shared_family_for_stem(file_stem) 
            documents.append(Document(page_content=content, metadata={ 
                'domain': domain_dir.name, 
                'source': str(file_path.relative_to(raw_dir)), 
                'section': section, 
                'doc_id': f'{domain_dir.name}:{file_stem}', 
                'file_stem': file_stem, 
                'shared_family': shared_family, 
                'knowledge_layer': 'domain', 
            })) 
    return documents
