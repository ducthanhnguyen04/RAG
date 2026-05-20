"""
Document Manager - Quản lý các tài liệu được lưu
Cho phép người dùng lưu, xóa, export các tài liệu phù hợp
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import streamlit as st


class DocumentManager:
    """
    Quản lý các tài liệu được lưu bởi người dùng
    """
    
    def __init__(self, save_dir: str = "data/saved_documents"):
        """
        Khởi tạo Document Manager
        
        Parameters:
        -----------
        save_dir : str
            Thư mục lưu trữ tài liệu
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._init_session_state()
    
    def _init_session_state(self):
        """Khởi tạo session state nếu chưa có"""
        # Chỉ tạo nếu chưa tồn tại, không reset nếu đã có
        if "saved_documents" not in st.session_state:
            st.session_state.saved_documents = []
        if "document_rating" not in st.session_state:
            st.session_state.document_rating = {}
    
    def save_document(
        self,
        content: str,
        title: str = "Untitled",
        source: str = "Unknown",
        relevance_score: float = 0.0,
        tags: Optional[List[str]] = None,
        query: Optional[str] = None
    ) -> Dict:
        """
        Lưu một tài liệu
        
        Parameters:
        -----------
        content : str
            Nội dung của tài liệu
        title : str
            Tiêu đề tài liệu
        source : str
            Nguồn gốc tài liệu
        relevance_score : float
            Điểm độ liên quan (0-1)
        tags : List[str], optional
            Các tag/nhãn cho tài liệu
        query : str, optional
            Câu hỏi liên quan
            
        Returns:
        --------
        Dict : Thông tin tài liệu đã lưu
        """
        
        doc_id = self._generate_id()
        
        document = {
            "id": doc_id,
            "title": title,
            "content": content,
            "source": source,
            "relevance_score": relevance_score,
            "tags": tags or [],
            "query": query,
            "saved_at": datetime.now().isoformat(),
            "rating": 0,
            "notes": ""
        }
        
        # Debug: Kiểm tra trạng thái trước save
        import sys
        print(f"[SAVE_DEBUG] Trước save: len={len(st.session_state.saved_documents)}", file=sys.stderr)
        print(f"[SAVE_DEBUG] Nội dung: title={title[:30]}, content_len={len(content)}", file=sys.stderr)
        
        # Lưu vào session state
        st.session_state.saved_documents.append(document)
        
        # Debug: Kiểm tra trạng thái sau save
        print(f"[SAVE_DEBUG] Sau save: len={len(st.session_state.saved_documents)}", file=sys.stderr)
        print(f"[SAVE_DEBUG] Documents list: {[d['title'] for d in st.session_state.saved_documents]}", file=sys.stderr)
        
        return document
    
    def get_all_saved_documents(self) -> List[Dict]:
        """Lấy tất cả tài liệu đã lưu"""
        return st.session_state.saved_documents
    
    def get_saved_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Lấy một tài liệu theo ID"""
        for doc in st.session_state.saved_documents:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Xóa một tài liệu theo ID
        
        Returns:
        --------
        bool : True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, doc in enumerate(st.session_state.saved_documents):
            if doc["id"] == doc_id:
                st.session_state.saved_documents.pop(i)
                return True
        return False
    
    def update_document_notes(self, doc_id: str, notes: str) -> bool:
        """Cập nhật ghi chú cho tài liệu"""
        doc = self.get_saved_document_by_id(doc_id)
        if doc:
            doc["notes"] = notes
            return True
        return False
    
    def rate_document(self, doc_id: str, rating: int) -> bool:
        """
        Đánh giá tài liệu (1-5 sao)
        
        Parameters:
        -----------
        doc_id : str
            ID của tài liệu
        rating : int
            Đánh giá (1-5)
        """
        doc = self.get_saved_document_by_id(doc_id)
        if doc:
            doc["rating"] = max(1, min(5, rating))
            return True
        return False
    
    def search_documents(
        self,
        query: str,
        search_in: str = "all"  # "title", "content", "tags", "all"
    ) -> List[Dict]:
        """
        Tìm kiếm tài liệu đã lưu
        
        Parameters:
        -----------
        query : str
            Chuỗi tìm kiếm
        search_in : str
            Tìm kiếm trong: "title", "content", "tags", "all"
        """
        results = []
        query_lower = query.lower()
        
        for doc in st.session_state.saved_documents:
            found = False
            
            if search_in in ["title", "all"]:
                if query_lower in doc["title"].lower():
                    found = True
            
            if search_in in ["content", "all"]:
                if query_lower in doc["content"].lower():
                    found = True
            
            if search_in in ["tags", "all"]:
                if any(query_lower in tag.lower() for tag in doc["tags"]):
                    found = True
            
            if found:
                results.append(doc)
        
        return results
    
    def filter_by_score(self, min_score: float = 0.0) -> List[Dict]:
        """Lọc tài liệu theo điểm tối thiểu"""
        return [
            doc for doc in st.session_state.saved_documents
            if doc["relevance_score"] >= min_score
        ]
    
    def filter_by_rating(self, min_rating: int = 1) -> List[Dict]:
        """Lọc tài liệu theo đánh giá tối thiểu"""
        return [
            doc for doc in st.session_state.saved_documents
            if doc.get("rating", 0) >= min_rating
        ]
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export các tài liệu đã lưu sang JSON
        
        Returns:
        --------
        str : Đường dẫn file hoặc JSON string
        """
        if filepath:
            filepath = self.save_dir / filepath
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.saved_documents, f, ensure_ascii=False, indent=2)
            return str(filepath)
        else:
            return json.dumps(st.session_state.saved_documents, ensure_ascii=False, indent=2)
    
    def export_to_markdown(self) -> str:
        """Export các tài liệu đã lưu sang Markdown"""
        markdown = "# 📚 Tài Liệu Đã Lưu\n\n"
        markdown += f"Ngày xuất: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown += f"Tổng cộng: {len(st.session_state.saved_documents)} tài liệu\n\n"
        
        for doc in st.session_state.saved_documents:
            markdown += f"## {doc['title']}\n\n"
            markdown += f"- **Nguồn:** {doc['source']}\n"
            markdown += f"- **Điểm Liên Quan:** {doc['relevance_score']:.2f}\n"
            markdown += f"- **Đánh Giá:** {'⭐' * doc.get('rating', 0)}\n"
            
            if doc.get('query'):
                markdown += f"- **Câu Hỏi:** {doc['query']}\n"
            
            if doc.get('tags'):
                markdown += f"- **Tags:** {', '.join(doc['tags'])}\n"
            
            markdown += f"- **Lưu Lúc:** {doc['saved_at']}\n\n"
            
            markdown += f"### Nội Dung\n{doc['content']}\n\n"
            
            if doc.get('notes'):
                markdown += f"### Ghi Chú\n{doc['notes']}\n\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    def export_to_csv(self) -> str:
        """Export tài liệu sang CSV"""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "ID", "Tiêu Đề", "Nguồn", "Điểm Liên Quan", "Đánh Giá",
            "Tags", "Câu Hỏi", "Lưu Lúc", "Ghi Chú"
        ])
        
        # Data
        for doc in st.session_state.saved_documents:
            writer.writerow([
                doc["id"],
                doc["title"],
                doc["source"],
                f"{doc['relevance_score']:.2f}",
                doc.get("rating", 0),
                ";".join(doc.get("tags", [])),
                doc.get("query", ""),
                doc["saved_at"],
                doc.get("notes", "")
            ])
        
        return output.getvalue()
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê về các tài liệu đã lưu"""
        docs = st.session_state.saved_documents
        
        if not docs:
            return {
                "total": 0,
                "avg_score": 0,
                "avg_rating": 0,
                "sources": [],
                "tags": []
            }
        
        total = len(docs)
        avg_score = sum(d.get("relevance_score", 0) for d in docs) / total
        avg_rating = sum(d.get("rating", 0) for d in docs) / total
        
        sources = list(set(d.get("source", "Unknown") for d in docs))
        all_tags = []
        for doc in docs:
            all_tags.extend(doc.get("tags", []))
        
        return {
            "total": total,
            "avg_score": avg_score,
            "avg_rating": avg_rating,
            "sources": sources,
            "tags": list(set(all_tags)),
            "by_source": {
                source: len([d for d in docs if d.get("source") == source])
                for source in sources
            }
        }
    
    def _generate_id(self) -> str:
        """Tạo ID duy nhất cho tài liệu"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_unique_sources(self) -> List[str]:
        """Lấy danh sách các nguồn duy nhất"""
        sources = set()
        for doc in st.session_state.saved_documents:
            sources.add(doc.get("source", "Unknown"))
        return sorted(list(sources))
    
    def get_all_tags(self) -> List[str]:
        """Lấy tất cả các tags"""
        tags = set()
        for doc in st.session_state.saved_documents:
            tags.update(doc.get("tags", []))
        return sorted(list(tags))
    
    def add_tags_to_document(self, doc_id: str, tags: List[str]) -> bool:
        """Thêm tags vào một tài liệu"""
        doc = self.get_saved_document_by_id(doc_id)
        if doc:
            doc["tags"] = list(set(doc.get("tags", []) + tags))
            return True
        return False
    
    def remove_tags_from_document(self, doc_id: str, tags: List[str]) -> bool:
        """Xóa tags khỏi một tài liệu"""
        doc = self.get_saved_document_by_id(doc_id)
        if doc:
            doc["tags"] = [t for t in doc.get("tags", []) if t not in tags]
            return True
        return False


def create_save_button_ui(doc_manager: DocumentManager, doc_content: str, doc_title: str, doc_source: str, doc_score: float, query: str = None) -> bool:
    """
    Tạo giao diện nút lưu tài liệu
    
    Returns:
    --------
    bool : True nếu người dùng lưu tài liệu
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Lưu Tài Liệu", key=f"save_{hash(doc_content) % 10**8}"):
            doc_manager.save_document(
                content=doc_content,
                title=doc_title,
                source=doc_source,
                relevance_score=doc_score,
                query=query
            )
            st.success("✅ Tài liệu đã được lưu!")
            return True
    
    with col2:
        st.write("")  # Placeholder
    
    with col3:
        st.write("")  # Placeholder
    
    return False


def display_saved_document(doc: Dict, doc_manager: DocumentManager):
    """
    Hiển thị một tài liệu đã lưu với các tùy chọn chỉnh sửa
    
    Parameters:
    -----------
    doc : Dict
        Tài liệu cần hiển thị
    doc_manager : DocumentManager
        Người quản lý tài liệu
    """
    # Header with title, source, and score
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"### {doc['title']}")
    
    with col2:
        st.markdown(f"**Điểm:** {doc['relevance_score']:.2f}")
    
    with col3:
        if doc.get('rating'):
            st.markdown(f"**Đánh Giá:** {'⭐' * doc['rating']}")
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"📌 **Nguồn:** {doc['source']}")
    
    with col2:
        st.markdown(f"🏷️ **Tags:** {', '.join(doc.get('tags', [])) if doc.get('tags') else 'Chưa có'}")
    
    with col3:
        st.markdown(f"⏰ **Lưu:** {doc['saved_at'][:10]}")
    
    # Content
    with st.expander("📖 Xem Nội Dung"):
        st.markdown(doc['content'])
    
    # Notes and editing
    col1, col2 = st.columns(2)
    
    with col1:
        new_notes = st.text_area(
            "Ghi Chú",
            value=doc.get('notes', ''),
            key=f"notes_{doc['id']}",
            height=100
        )
        if new_notes != doc.get('notes', ''):
            doc_manager.update_document_notes(doc['id'], new_notes)
    
    with col2:
        # Rating
        new_rating = st.slider(
            "Đánh Giá",
            min_value=0,
            max_value=5,
            value=doc.get('rating', 0),
            key=f"rating_{doc['id']}"
        )
        if new_rating != doc.get('rating', 0):
            doc_manager.rate_document(doc['id'], new_rating)
    
    # Actions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Xóa", key=f"delete_{doc['id']}"):
            doc_manager.delete_document(doc['id'])
            st.success("✅ Tài liệu đã được xóa")
            st.rerun()
    
    with col2:
        st.write("")  # Placeholder
    
    with col3:
        st.write("")  # Placeholder
    
    with col4:
        st.write("")  # Placeholder
