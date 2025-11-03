import os
import json
import psycopg2
import psycopg2.errors
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from typing import List, Dict, Any, Optional

from app.config import (
    EMBEDDING_MODEL, OLLAMA_HOST, DATA_DIR, EMBEDDING_DIM,
    DEFAULT_TABLE_NAME, DEFAULT_POSTGRES_HOST, DEFAULT_POSTGRES_PORT,
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    DEFAULT_RETRIEVAL_LIMIT, DEFAULT_DOC_LIMIT, DEFAULT_DOC_LIMIT_SOURCE,
    DEFAULT_DOC_LIMIT_METADATA, DEFAULT_DOC_LIMIT_PATTERN
)
from app.tracing import trace

class VectorStorageService:
    def __init__(self,
                 default_json_path: str = None,
                 db_table_name: str = None,
                 embedding_dimension: int = EMBEDDING_DIM):
        """
        Initialize Vector Database Service
        
        Args:
            default_json_path: Path to JSON file with chunks
            db_table_name: Database table name
            embedding_dimension: Embedding dimensions
        """
        self._load_environment_variables()
        self._setup_database_connection_details()
        self.default_json_path = default_json_path
        self.table_name = db_table_name or DEFAULT_TABLE_NAME
        self.embed_dim = embedding_dimension
        self.conn = None
        self.embed_model = None
        self.vector_store = None
        self.index = None
        
    def _load_environment_variables(self) -> None:
        """Load environment variables"""
        load_dotenv(dotenv_path='../.env')
        load_dotenv(dotenv_path='/app/.env')
        load_dotenv()
        
    def _setup_database_connection_details(self) -> None:
        """Setup database credentials"""
        self.database = POSTGRES_DB
        self.user = POSTGRES_USER
        self.password = POSTGRES_PASSWORD
        self.host = DEFAULT_POSTGRES_HOST
        self.port = DEFAULT_POSTGRES_PORT
        
        if not all([self.database, self.user, self.password]):
            raise ValueError(
                f"Missing database credentials. "
                f"DB: {self.database}, User: {self.user}, "
                f"Password: {'SET' if self.password else 'NOT SET'}"
            )
    
    def _connect_to_db(self) -> None:
        """Connect to database"""
        self.conn = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        self.conn.autocommit = True
        
    def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist"""
        if not self.conn:
            self._connect_to_db()
            
        with self.conn.cursor() as c:
            try:
                c.execute(f"CREATE DATABASE {self.database}")
                print(f"Database {self.database} created")
            except psycopg2.errors.DuplicateDatabase:
                print(f"Database {self.database} already exists")
    
    def get_available_databases(self) -> List[str]:
        """List all databases"""
        if not self.conn:
            self._connect_to_db()
            
        with self.conn.cursor() as c:
            c.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
            dbs = [db[0] for db in c.fetchall()]
            return dbs
    
    def _initialize_embedding_model(self) -> None:
        """Initialize embedding model"""
        self.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_HOST
        )
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store"""
        self.vector_store = PGVectorStore.from_params(
            database=self.database,
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
            table_name=self.table_name,
            embed_dim=self.embed_dim,
        )
    
    def _initialize_vector_index(self) -> None:
        """Initialize index"""
        if not self.vector_store:
            self._initialize_vector_store()
            
        if not self.embed_model:
            self._initialize_embedding_model()
            
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
            show_progress=True,
            embed_model=self.embed_model,
        )
    
    def _load_chunks_from_json(self, json_file_path: str = None) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON file
        
        Args:
            json_file_path: Path to JSON file (if not provided during initialization)
            
        Returns:
            List of chunks
        """
        file_path = json_file_path or self.default_json_path
        if not file_path:
            raise ValueError("JSON file path not provided")
        
        data = None
        # Try different paths if only filename is provided
        possible_paths = [
            file_path,
            f"clear_docs/{file_path}",
            f"{DATA_DIR}/{file_path}",
            f"/app/data/{file_path}",
            f"../data/{file_path}"
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Loaded data from: {path}")
                    break
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {path}: {e}")
        
        if data is None:
            raise FileNotFoundError(
                f"Could not find JSON file. Tried: {', '.join(possible_paths)}"
            )
        
        chunks = []
        if "chunks" in data:
            for chunk in data["chunks"]:
                chunks.append(chunk)
        else:
            chunks = data if isinstance(data, list) else [data]
            
        return chunks
    
    def _create_text_nodes(self, chunks: List[Dict[str, Any]], 
                               source_name: str = None) -> List[TextNode]:
        """
        Create TextNode from chunks
        
        Args:
            chunks: List of chunks
            source_name: Source name
            
        Returns:
            List of TextNode
        """
        if not source_name and self.default_json_path:
            source_name = os.path.basename(self.default_json_path).replace('.json', '')
        else:
            source_name = source_name or "unknown_source"
            
        nodes = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get('text', chunk.get('content', str(chunk)))
                chunk_id = chunk.get('id', f"{source_name}_{i}")
            else:
                text = str(chunk)
                chunk_id = f"{source_name}_{i}"
                
            node = TextNode(
                text=text,
                metadata={
                    'id': chunk_id,
                    'source': source_name,
                    'source_file': self.default_json_path or source_name,
                    'document_id': chunk_id,
                    'chunk_index': i
                }
            )
            nodes.append(node)
            
        return nodes
    
    def insert_nodes(self, chunks: List[Dict[str, Any]], 
                     source_name: str = None) -> int:
        """
        Insert chunks into database
        
        Args:
            chunks: List of chunks
            source_name: Source name
            
        Returns:
            Number of inserted documents
        """
        if not self.index:
            self._initialize_vector_index()
            
        nodes = self._create_text_nodes(chunks, source_name)
        self.index.insert_nodes(nodes)
        
        return len(nodes)
    
    def load_and_insert_from_json(self, json_file_path: str = None, 
                                 source_name: str = None) -> int:
        """
        Complete process: load JSON -> create chunks -> insert to DB
        
        Args:
            json_file_path: Path to JSON file
            source_name: Source name
            
        Returns:
            Number of inserted documents
        """
        if not self.index:
            self._initialize_database_and_index()
        
        if json_file_path:
            self.default_json_path = json_file_path
            
        chunks = self._load_chunks_from_json()
        count = self.insert_nodes(chunks, source_name)
        
        print(f"Successfully inserted {count} chunks from {self.default_json_path}")
        return count
    
    def get_total_document_count(self) -> int:
        """Get document count in database"""
        if not self.conn:
            self._connect_to_db()
            
        with self.conn.cursor() as c:
            c.execute(f"SELECT COUNT(*) FROM data_{self.table_name}")
            count = c.fetchone()[0]
            return count
    
    def _initialize_database_and_index(self):
        """Private helper to connect, create DB, and init index."""
        self._connect_to_db()
        self._ensure_database_exists()
        
        dbs = self.get_available_databases()
        print("Available databases:")
        for db in dbs:
            print(f"- {db}")
        
        self._initialize_embedding_model()
        self._initialize_vector_store()
        self._initialize_vector_index()
    
    def setup_and_ingest_pipeline(self, json_file_path: str = None) -> int:
        """
        Complete pipeline setup and data insertion
        
        Args:
            json_file_path: Path to JSON file
            
        Returns:
            Number of inserted documents
        """
        print("Setting up vector database pipeline...")
        
        self._initialize_database_and_index()
        
        count = self.load_and_insert_from_json(json_file_path)
        
        final_count = self.get_total_document_count()
        print(f"{count} docs has been inserted. Total documents in database: {final_count}")
        
        return final_count
    
    
    def search_similar_documents(self, query: str, limit: int = DEFAULT_RETRIEVAL_LIMIT) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity
            
        Args:
            query: Search query
            limit: Number of results to return
                
        Returns:
            List of similar documents with metadata
        """
        if not self.index:
            self._initialize_vector_index()
        retriever = self.index.as_retriever(similarity_top_k=limit)

        with trace("rag.retrieve", {"query": query[:100], "limit": limit}) as span:
            response_nodes = retriever.retrieve(query)
            if span:
                span.set_attribute("documents_found", len(response_nodes))
        
        return response_nodes

    def fetch_all_documents(self, limit: int = DEFAULT_DOC_LIMIT) -> List[Dict[str, Any]]:
        """
        Get all documents from the database
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of all documents
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                LIMIT %s
            """, (limit,))
            
            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })
            
            return documents

    def fetch_documents_by_source(self, source: str, limit: int = DEFAULT_DOC_LIMIT_SOURCE) -> List[Dict[str, Any]]:
        """
        Get documents by source name
        
        Args:
            source: Source name to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of documents from specified source
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                WHERE metadata->>'source' = %s 
                LIMIT %s
            """, (source, limit))
            
            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })
            
            return documents

    def fetch_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific document by ID
        
        Args:
            doc_id: Document ID to search for
            
        Returns:
            Document data if found, None otherwise
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                WHERE metadata->>'id' = %s
            """, (doc_id,))
            
            result = c.fetchone()
            if result:
                return {
                    'text': result[0],
                    'metadata': result[1]
                }
            return None

    def fetch_documents_by_metadata(self, metadata_key: str, metadata_value: str, 
                                    limit: int = DEFAULT_DOC_LIMIT_METADATA) -> List[Dict[str, Any]]:
        """
        Search documents by metadata key-value pair
        
        Args:
            metadata_key: Metadata key to search
            metadata_value: Metadata value to match
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata_
                FROM data_{self.table_name} 
                WHERE metadata_->>%s = %s 
                LIMIT %s
            """, (metadata_key, metadata_value, limit))
            
            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })
            
            return documents

    def fetch_chunks_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document ID to get chunks for
            
        Returns:
            List of chunks belonging to the document
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                WHERE metadata->>'document_id' = %s 
                ORDER BY (metadata->>'chunk_index')::int
            """, (document_id,))
            
            chunks = []
            for row in c.fetchall():
                chunks.append({
                    'text': row[0],
                    'metadata': row[1]
                })
            
            return chunks

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with various statistics
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"SELECT COUNT(*) FROM data_{self.table_name}")
            total_docs = c.fetchone()[0]
            
            c.execute(f"""
                SELECT metadata->>'source', COUNT(*) 
                FROM data_{self.table_name} 
                GROUP BY metadata->>'source'
            """)
            sources = {row[0]: row[1] for row in c.fetchall()}
            
            c.execute(f"SELECT AVG(LENGTH(text)) FROM data_{self.table_name}")
            avg_length = c.fetchone()[0]
            
            return {
                'total_documents': total_docs,
                'sources_distribution': sources,
                'average_text_length': float(avg_length) if avg_length else 0,
                'table_name': self.table_name
            }

    def delete_document_by_id(self, doc_id: str) -> bool:
        """
        Delete document by ID
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                DELETE FROM data_{self.table_name} 
                WHERE metadata->>'id' = %s
            """, (doc_id,))
            
            return c.rowcount > 0

    def delete_documents_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source
        
        Args:
            source: Source name to delete
            
        Returns:
            Number of deleted documents
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                DELETE FROM data_{self.table_name} 
                WHERE metadata->>'source' = %s
            """, (source,))
            
            return c.rowcount
        
    def fetch_document_by_source_file(self, source_file: str) -> Optional[Dict[str, Any]]:
        """
        Get specific document by source file
        
        Args:
            source_file: Source file to search for
            
        Returns:
            Document data if found, None otherwise
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                WHERE metadata->>'source_file' = %s
            """, (source_file,))
            
            result = c.fetchone()
            if result:
                return {
                    'text': result[0],
                    'metadata': result[1]
                }
            return None

    def fetch_chunks_by_source_file(self, source_file: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific source file
        
        Args:
            source_file: Source file to get chunks for
            
        Returns:
            List of chunks belonging to the source file
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                WHERE metadata->>'source_file' = %s 
                ORDER BY (metadata->>'chunk_index')::int
            """, (source_file,))
            
            chunks = []
            for row in c.fetchall():
                chunks.append({
                    'text': row[0],
                    'metadata': row[1]
                })
            
            return chunks

    def delete_documents_by_source_file(self, source_file: str) -> bool:
        """
        Delete document by source file
        
        Args:
            source_file: Source file to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                DELETE FROM data_{self.table_name} 
                WHERE metadata->>'source_file' = %s
            """, (source_file,))
            
            return c.rowcount > 0

    def fetch_documents_by_source_file_pattern(self, pattern: str, limit: int = DEFAULT_DOC_LIMIT_PATTERN) -> List[Dict[str, Any]]:
        """
        Get documents by source file pattern (LIKE search)
        
        Args:
            pattern: SQL LIKE pattern (e.g., '%.json', 'test%')
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.table_name} 
                WHERE metadata->>'source_file' LIKE %s 
                LIMIT %s
            """, (pattern, limit))
            
            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })
            
            return documents
    
    def close_connection(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
        
    def debug_print_table_structure(self) -> None:
        """Debug method to check table structure"""
        if not self.conn:
            self._connect_to_db()
        
        with self.conn.cursor() as c:
            c.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = %s
            """, (f"data_{self.table_name}",))
            
            table_exists = c.fetchone()
            print(f"Table data_{self.table_name} exists: {bool(table_exists)}")
            
            if table_exists:
                c.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (f"data_{self.table_name}",))
                
                columns = c.fetchall()
                print("Table columns:")
                for col in columns:
                    print(f"- {col[0]}: {col[1]}")
                
                c.execute(f"SELECT * FROM data_{self.table_name} LIMIT 1")
                sample = c.fetchone()
                if sample:
                    print("Sample row:", sample)
                    
    def drop_vector_table(self) -> bool:
        """
        Drop the entire table (more destructive than clear)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            self._connect_to_db()
        
        try:
            with self.conn.cursor() as c:
                c.execute(f"DROP TABLE IF EXISTS data_{self.table_name}")
                print(f"Table dropped: data_{self.table_name}")
                return True
        except Exception as e:
            print(f"Error dropping table: {e}")
            return False
    
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()