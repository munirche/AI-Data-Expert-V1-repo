"""
Expert Learning System V2 - Generic RAG Engine

A domain-agnostic engine that learns from expert annotations.
Use case specifics are defined in config.json, not in this code.
"""

import json
import os
import uuid
from datetime import datetime
from functools import cached_property
from typing import List, Dict, Any, Optional

import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import register, TextEmbeddingFunction, get_registry
from google import genai


# =============================================================================
# EMBEDDING FUNCTION
# =============================================================================

@register("gemini-genai-v2")
class GeminiEmbedding(TextEmbeddingFunction):
    """Custom embedding function using google-genai."""

    model_name: str = "text-embedding-004"

    @cached_property
    def _client(self):
        api_key = os.environ.get('GEMINI_API_KEY')
        return genai.Client(api_key=api_key)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def ndims(self) -> int:
        """Return embedding dimensions."""
        return 768


# Initialize embedding function
gemini_embed = get_registry().get("gemini-genai-v2").create()


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

class Annotation(LanceModel):
    """Schema for expert annotations stored in LanceDB."""

    annotation_id: str
    record_id: str
    timestamp: str
    record_data: str  # JSON string of original record
    summary: str
    analysis: str
    risk_assessment: str
    patterns: str  # JSON string
    recommended_actions: str  # JSON string
    additional_tests: str  # JSON string
    tags: str  # Comma-separated
    text: str = gemini_embed.SourceField()  # Field to embed
    vector: Vector(gemini_embed.ndims()) = gemini_embed.VectorField(default=None)


# =============================================================================
# ENGINE CLASS
# =============================================================================

class ExpertLearningEngine:
    """Generic RAG engine for expert learning system."""

    def __init__(self, config_path: str = "./config.json"):
        """Initialize engine with configuration."""
        self.config = self._load_config(config_path)
        self.db = lancedb.connect(self.config["database_path"])
        self.table_name = "expert_annotations"
        self.table = None
        self._init_table()

        # Initialize LLM client
        api_key = os.environ.get('GEMINI_API_KEY')
        self.llm_client = genai.Client(api_key=api_key)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _init_table(self):
        """Initialize or open the database table."""
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Table doesn't exist yet, will be created on first insert
            pass

    # -------------------------------------------------------------------------
    # CORPUS OPERATIONS
    # -------------------------------------------------------------------------

    def load_corpus(self) -> pd.DataFrame:
        """Load the corpus CSV file."""
        return pd.read_csv(self.config["corpus_path"])

    def get_corpus_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a single record from corpus by ID."""
        df = self.load_corpus()
        record_id_field = self.config["record_id_field"]

        # Convert to string for comparison
        df[record_id_field] = df[record_id_field].astype(str)
        matches = df[df[record_id_field] == str(record_id)]

        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def get_corpus_range(self, start: int, end: int) -> List[Dict[str, Any]]:
        """Get a range of records from corpus (by row index)."""
        df = self.load_corpus()
        return df.iloc[start:end].to_dict('records')

    # -------------------------------------------------------------------------
    # DATABASE OPERATIONS
    # -------------------------------------------------------------------------

    def store_annotation(
        self,
        record_id: str,
        record_data: Dict[str, Any],
        summary: str,
        analysis: str,
        risk_assessment: str = "",
        patterns: List[Dict] = None,
        recommended_actions: List[str] = None,
        additional_tests: List[str] = None,
        tags: List[str] = None
    ) -> str:
        """Store an expert annotation in the database."""

        annotation_id = str(uuid.uuid4())[:8]

        # Prepare the text field for embedding (summary + analysis)
        text_for_embedding = f"{summary}\n\n{analysis}"

        record = Annotation(
            annotation_id=annotation_id,
            record_id=str(record_id),
            timestamp=datetime.now().isoformat(),
            record_data=json.dumps(record_data),
            summary=summary,
            analysis=analysis,
            risk_assessment=risk_assessment or "",
            patterns=json.dumps(patterns or []),
            recommended_actions=json.dumps(recommended_actions or []),
            additional_tests=json.dumps(additional_tests or []),
            tags=",".join(tags or []),
            text=text_for_embedding
        )

        # Create table on first insert
        if self.table is None:
            self.table = self.db.create_table(self.table_name, schema=Annotation)

        self.table.add([record])
        return annotation_id

    def load_from_corpus(self, record_id: str) -> Optional[str]:
        """Load a record from corpus and store its annotation."""
        record = self.get_corpus_record(record_id)
        if record is None:
            return None

        # Extract data fields vs annotation fields
        data_fields = self.config["data_fields"]
        record_id_field = self.config["record_id_field"]

        record_data = {k: record[k] for k in data_fields if k in record}
        record_data[record_id_field] = record[record_id_field]

        # Parse patterns from JSON string if present
        patterns = []
        if 'patterns' in record and record['patterns']:
            try:
                patterns = json.loads(record['patterns'])
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse actions and tests
        actions = []
        if 'recommended_actions' in record and record['recommended_actions']:
            actions = [a.strip() for a in str(record['recommended_actions']).split(';')]

        tests = []
        if 'additional_tests' in record and record['additional_tests']:
            tests = [t.strip() for t in str(record['additional_tests']).split(';') if t.strip()]

        # Parse tags
        tags = []
        if 'tags' in record and record['tags']:
            tags = [t.strip() for t in str(record['tags']).split(',')]

        return self.store_annotation(
            record_id=str(record[record_id_field]),
            record_data=record_data,
            summary=record.get('summary', ''),
            analysis=record.get('analysis', ''),
            risk_assessment=record.get('risk_assessment', ''),
            patterns=patterns,
            recommended_actions=actions,
            additional_tests=tests,
            tags=tags
        )

    def load_n_from_corpus(self, n: int) -> List[str]:
        """Load first N records from corpus into database."""
        df = self.load_corpus()
        record_id_field = self.config["record_id_field"]

        loaded = []
        for i in range(min(n, len(df))):
            record_id = str(df.iloc[i][record_id_field])
            ann_id = self.load_from_corpus(record_id)
            if ann_id:
                loaded.append(ann_id)

        return loaded

    def list_annotations(self, limit: int = None, tag: str = None) -> List[Dict[str, Any]]:
        """List all annotations in the database."""
        if self.table is None:
            return []

        df = self.table.to_pandas()

        if tag:
            df = df[df['tags'].str.contains(tag, na=False)]

        if limit:
            df = df.head(limit)

        results = []
        for _, row in df.iterrows():
            results.append({
                'annotation_id': row['annotation_id'],
                'record_id': row['record_id'],
                'summary': row['summary'],
                'risk_assessment': row['risk_assessment'],
                'tags': row['tags'],
                'timestamp': row['timestamp']
            })

        return results

    def get_annotation(self, annotation_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a specific annotation."""
        if self.table is None:
            return None

        df = self.table.to_pandas()
        matches = df[df['annotation_id'] == annotation_id]

        if matches.empty:
            return None

        row = matches.iloc[0]
        return {
            'annotation_id': row['annotation_id'],
            'record_id': row['record_id'],
            'timestamp': row['timestamp'],
            'record_data': json.loads(row['record_data']),
            'summary': row['summary'],
            'analysis': row['analysis'],
            'risk_assessment': row['risk_assessment'],
            'patterns': json.loads(row['patterns']),
            'recommended_actions': json.loads(row['recommended_actions']),
            'additional_tests': json.loads(row['additional_tests']),
            'tags': row['tags'].split(',') if row['tags'] else []
        }

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar annotations."""
        if self.table is None:
            return []

        results = self.table.search(query).limit(limit).to_list()

        similar = []
        for row in results:
            similar.append({
                'annotation_id': row['annotation_id'],
                'record_id': row['record_id'],
                'summary': row['summary'],
                'analysis': row['analysis'],
                'risk_assessment': row['risk_assessment'],
                'similarity_score': 1 / (1 + row.get('_distance', 0)),
                'tags': row['tags']
            })

        return similar

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        corpus_df = self.load_corpus()
        corpus_count = len(corpus_df)

        if self.table is None:
            return {
                'total_annotations': 0,
                'corpus_total': corpus_count,
                'corpus_loaded': 0,
                'corpus_remaining': corpus_count,
                'risk_distribution': {},
                'top_tags': []
            }

        df = self.table.to_pandas()
        total = len(df)

        # Risk distribution
        risk_counts = df['risk_assessment'].value_counts().to_dict()

        # Top tags
        all_tags = []
        for tags in df['tags']:
            if tags:
                all_tags.extend([t.strip() for t in tags.split(',')])

        from collections import Counter
        tag_counts = Counter(all_tags).most_common(5)

        return {
            'total_annotations': total,
            'corpus_total': corpus_count,
            'corpus_loaded': total,
            'corpus_remaining': corpus_count - total,
            'risk_distribution': risk_counts,
            'top_tags': tag_counts
        }

    def reset_database(self) -> bool:
        """Clear all annotations from the database."""
        try:
            self.db.drop_table(self.table_name)
            self.table = None
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # AI ANALYSIS
    # -------------------------------------------------------------------------

    def retrieve_similar(self, record_data: Dict[str, Any], n: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar past annotations for a record."""
        # Create a query from the record data
        query_parts = []
        for key, value in record_data.items():
            query_parts.append(f"{key}: {value}")
        query = "\n".join(query_parts)

        return self.search_similar(query, limit=n)

    def generate_analysis(
        self,
        record_data: Dict[str, Any],
        similar_analyses: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate AI analysis for a record."""

        # Format examples
        examples_text = ""
        if similar_analyses:
            for i, ex in enumerate(similar_analyses, 1):
                examples_text += f"\nExample {i} ({ex['similarity_score']:.0%} similar):\n"
                examples_text += f"Summary: {ex['summary']}\n"
                examples_text += f"Analysis: {ex['analysis']}\n"
                examples_text += f"Risk: {ex['risk_assessment']}\n"
        else:
            examples_text = "No similar past analyses available."

        # Format record
        record_text = "\n".join([f"{k}: {v}" for k, v in record_data.items()])

        # Build prompt from template
        prompt = self.config["ai_prompt_template"].format(
            examples=examples_text,
            record=record_text
        )

        # Call LLM
        response = self.llm_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return {
            'analysis': response.text,
            'similar_count': len(similar_analyses) if similar_analyses else 0,
            'timestamp': datetime.now().isoformat()
        }
