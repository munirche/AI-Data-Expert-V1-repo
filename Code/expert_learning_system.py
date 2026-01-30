"""
Expert Learning System - RAG Implementation
A practical system that learns from expert annotations and produces analyses

This implementation shows:
1. How experts annotate datasets and document findings
2. How the system stores annotations in a vector database
3. How to retrieve similar past analyses for new datasets
4. How to generate AI analyses based on expert examples
5. How to incorporate expert feedback back into the system
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import anthropic  # or use openai

class ExpertLearningSystem:
    """
    Main system class that manages the learning loop
    """
    
    def __init__(self, anthropic_api_key: str):
        # Initialize ChromaDB (lightweight vector database)
        self.chroma_client = chromadb.Client()
        
        # Create collection for storing annotations
        self.collection = self.chroma_client.get_or_create_collection(
            name="expert_annotations",
            metadata={"description": "Expert analysis knowledge base"}
        )
        
        # Initialize LLM client (using Claude as example)
        self.claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
    def store_expert_annotation(
        self,
        dataset_summary: str,
        expert_analysis: str,
        patterns_found: List[Dict[str, Any]],
        tags: List[str],
        expert_id: str
    ) -> str:
        """
        Store an expert's annotation in the knowledge base
        
        Args:
            dataset_summary: Summary of the dataset analyzed
            expert_analysis: Expert's natural language findings
            patterns_found: List of specific patterns detected
            tags: Categorization tags (e.g., ["anomaly", "revenue"])
            expert_id: Identifier for the expert
            
        Returns:
            annotation_id: Unique ID for this annotation
        """
        
        annotation_id = str(uuid.uuid4())
        
        # Create the annotation record
        annotation = {
            "annotation_id": annotation_id,
            "timestamp": datetime.now().isoformat(),
            "expert_id": expert_id,
            "dataset_summary": dataset_summary,
            "expert_analysis": expert_analysis,
            "patterns_found": patterns_found,
            "tags": tags
        }
        
        # Store in vector database
        # The embedding is automatically created from the text
        self.collection.add(
            ids=[annotation_id],
            documents=[expert_analysis],  # This gets embedded
            metadatas=[{
                "dataset_summary": dataset_summary,
                "tags": ",".join(tags),
                "expert_id": expert_id,
                "timestamp": annotation["timestamp"],
                "patterns": json.dumps(patterns_found)
            }]
        )
        
        print(f"✓ Stored annotation {annotation_id}")
        return annotation_id
    
    def retrieve_similar_analyses(
        self,
        dataset_summary: str,
        n_results: int = 5,
        tag_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past expert analyses
        
        Args:
            dataset_summary: Summary of new dataset to analyze
            n_results: How many similar examples to retrieve
            tag_filter: Optional tag to filter by
            
        Returns:
            List of similar past analyses
        """
        
        # Build query filter if tag specified
        where_filter = None
        if tag_filter:
            where_filter = {"tags": {"$contains": tag_filter}}
        
        # Query the vector database
        results = self.collection.query(
            query_texts=[dataset_summary],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        similar_analyses = []
        for i in range(len(results['ids'][0])):
            similar_analyses.append({
                "id": results['ids'][0][i],
                "expert_analysis": results['documents'][0][i],
                "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "metadata": results['metadatas'][0][i]
            })
        
        return similar_analyses
    
    def generate_ai_analysis(
        self,
        dataset_summary: str,
        dataset_details: str,
        retrieve_similar: bool = True
    ) -> Dict[str, Any]:
        """
        Generate AI analysis based on expert examples
        
        Args:
            dataset_summary: Brief summary of dataset
            dataset_details: Detailed data or description
            retrieve_similar: Whether to use RAG (recommended)
            
        Returns:
            Dictionary containing the AI analysis and metadata
        """
        
        similar_analyses = []
        if retrieve_similar:
            similar_analyses = self.retrieve_similar_analyses(
                dataset_summary,
                n_results=3
            )
        
        # Build the prompt with retrieved examples
        prompt = self._build_analysis_prompt(
            dataset_summary,
            dataset_details,
            similar_analyses
        )
        
        # Call Claude API
        response = self.claude_client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        ai_analysis = response.content[0].text
        
        return {
            "analysis": ai_analysis,
            "similar_examples_used": len(similar_analyses),
            "confidence": self._estimate_confidence(similar_analyses),
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_analysis_prompt(
        self,
        dataset_summary: str,
        dataset_details: str,
        similar_analyses: List[Dict[str, Any]]
    ) -> str:
        """Build the prompt for the LLM"""
        
        prompt = """You are an expert data analyst. Your task is to analyze a new dataset based on examples from previous expert analyses.

"""
        
        # Add similar examples if available
        if similar_analyses:
            prompt += "Here are similar analyses performed by experts in the past:\n\n"
            
            for i, example in enumerate(similar_analyses, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Dataset: {example['metadata']['dataset_summary']}\n"
                prompt += f"Expert Analysis: {example['expert_analysis']}\n"
                prompt += f"Patterns Found: {example['metadata']['patterns']}\n\n"
            
            prompt += "---\n\n"
        
        # Add the new dataset to analyze
        prompt += f"""Now analyze this NEW dataset:

Dataset Summary: {dataset_summary}

Dataset Details:
{dataset_details}

Provide a comprehensive analysis following the style and depth of the expert examples above. Include:
1. Key patterns or anomalies detected
2. Statistical insights
3. Potential explanations or hypotheses
4. Recommendations for further investigation

Analysis:"""
        
        return prompt
    
    def _estimate_confidence(self, similar_analyses: List[Dict[str, Any]]) -> float:
        """
        Estimate confidence based on similarity of retrieved examples
        Higher similarity to past examples = higher confidence
        """
        if not similar_analyses:
            return 0.3  # Low confidence with no examples
        
        # Average the similarity scores
        avg_similarity = sum(
            ex['similarity_score'] for ex in similar_analyses
        ) / len(similar_analyses)
        
        return min(avg_similarity * 1.2, 0.95)  # Cap at 95%
    
    def incorporate_feedback(
        self,
        ai_analysis: str,
        expert_corrections: str,
        dataset_summary: str,
        tags: List[str],
        expert_id: str
    ) -> str:
        """
        When expert corrects AI analysis, store the corrected version
        This improves future analyses
        """
        
        corrected_analysis = f"""AI Generated: {ai_analysis}

Expert Corrections and Improvements:
{expert_corrections}"""
        
        # Store as a new expert annotation
        return self.store_expert_annotation(
            dataset_summary=dataset_summary,
            expert_analysis=corrected_analysis,
            patterns_found=[{"type": "AI_corrected", "note": "This was an AI analysis corrected by expert"}],
            tags=tags + ["ai_corrected"],
            expert_id=expert_id
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        
        total_annotations = self.collection.count()
        
        # Could add more sophisticated metrics here
        return {
            "total_expert_annotations": total_annotations,
            "estimated_accuracy": min(0.5 + (total_annotations * 0.002), 0.95),
            "annotations_needed_for_90_percent": max(0, 200 - total_annotations)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_workflow():
    """
    Demonstrates the complete workflow:
    1. Expert annotates several datasets
    2. System learns from annotations
    3. AI analyzes new dataset
    4. Expert provides feedback
    5. System improves
    """
    
    # Initialize system
    system = ExpertLearningSystem(
        anthropic_api_key="your-api-key-here"
    )
    
    print("=" * 70)
    print("PHASE 1: Expert Annotates Initial Examples")
    print("=" * 70)
    
    # Expert analyzes first dataset
    annotation_1 = system.store_expert_annotation(
        dataset_summary="Q1-Q4 2023 revenue data by region",
        expert_analysis="""
        Key Findings:
        1. Significant anomaly in Q3 - revenue dropped 15% across all regions
        2. Investigation revealed system outage during migration
        3. Q4 shows recovery with 8% growth
        4. West region consistently outperforms (22% above average)
        5. Seasonality pattern: Q2 and Q4 are peak quarters
        
        Recommendations:
        - Exclude Q3 from baseline calculations
        - Focus growth investments in West region strategy
        - Prepare for seasonal spikes in Q2/Q4
        """,
        patterns_found=[
            {"type": "anomaly", "location": "Q3", "severity": "high"},
            {"type": "trend", "location": "West_region", "direction": "positive"},
            {"type": "seasonality", "period": "quarterly", "peaks": ["Q2", "Q4"]}
        ],
        tags=["revenue", "anomaly", "seasonality"],
        expert_id="analyst_jane"
    )
    
    # Expert analyzes second dataset
    annotation_2 = system.store_expert_annotation(
        dataset_summary="Customer churn data - Jan-Dec 2023",
        expert_analysis="""
        Key Findings:
        1. Churn rate increased from 5% to 12% between March-May
        2. Root cause: competitor launched similar product at 30% lower price
        3. High-value customers (>$10k/year) showed lower churn (3%)
        4. Most churn occurred in 30-60 day window after competitor launch
        
        Recommendations:
        - Implement retention program for mid-tier customers
        - Price adjustment analysis for competitive positioning
        - Enhanced value communication to at-risk segments
        """,
        patterns_found=[
            {"type": "anomaly", "location": "March-May", "severity": "high"},
            {"type": "correlation", "variables": ["competitor_launch", "churn_spike"]},
            {"type": "segmentation", "segment": "high_value", "behavior": "retention"}
        ],
        tags=["churn", "customer_behavior", "anomaly"],
        expert_id="analyst_jane"
    )
    
    # Expert analyzes third dataset  
    annotation_3 = system.store_expert_annotation(
        dataset_summary="Website traffic analysis - 2023 annual",
        expert_analysis="""
        Key Findings:
        1. Mobile traffic grew 45% while desktop declined 12%
        2. Conversion rate on mobile (1.2%) significantly lower than desktop (3.8%)
        3. Bounce rate spike in July correlated with site redesign
        4. Organic search traffic up 67% following SEO improvements
        
        Recommendations:
        - Prioritize mobile conversion optimization
        - Review July redesign for usability issues
        - Continue SEO strategy - strong ROI evident
        """,
        patterns_found=[
            {"type": "trend", "location": "mobile_traffic", "direction": "up", "magnitude": "45%"},
            {"type": "trend", "location": "desktop_traffic", "direction": "down", "magnitude": "12%"},
            {"type": "anomaly", "location": "July_bounce_rate", "cause": "redesign"}
        ],
        tags=["traffic", "conversion", "mobile"],
        expert_id="analyst_jane"
    )
    
    print(f"✓ Stored {3} expert annotations\n")
    
    # Check system stats
    stats = system.get_system_stats()
    print(f"System Stats: {json.dumps(stats, indent=2)}\n")
    
    print("=" * 70)
    print("PHASE 2: AI Analyzes New Dataset Using Expert Examples")
    print("=" * 70)
    
    # New dataset arrives - AI analyzes it
    new_dataset_summary = "Q1 2024 revenue data by region"
    new_dataset_details = """
    Region | Q1_2024_Revenue | Q4_2023_Revenue | YoY_Change
    -------|----------------|-----------------|------------
    East   | $2.3M          | $2.1M           | +9.5%
    West   | $3.8M          | $3.5M           | +8.6%
    North  | $1.9M          | $2.0M           | -5.0%
    South  | $2.1M          | $1.8M           | +16.7%
    
    Notes: 
    - North region shows unexpected decline
    - South region strong growth after new sales team
    """
    
    print("Analyzing new dataset...")
    ai_result = system.generate_ai_analysis(
        dataset_summary=new_dataset_summary,
        dataset_details=new_dataset_details,
        retrieve_similar=True
    )
    
    print(f"\nAI Analysis (Confidence: {ai_result['confidence']:.2%}):")
    print("-" * 70)
    print(ai_result['analysis'])
    print("-" * 70)
    print(f"\n✓ Used {ai_result['similar_examples_used']} similar expert examples")
    
    print("\n" + "=" * 70)
    print("PHASE 3: Expert Reviews and Provides Feedback")
    print("=" * 70)
    
    expert_corrections = """
    AI analysis was generally good, but missed a few points:
    
    Additional Insights:
    - North region decline is actually due to large client contract ending (planned)
    - Should note that West region maintaining strong performance is noteworthy given overall market softness
    - South region growth is impressive but should be monitored for sustainability
    
    Recommendations should also include:
    - Strategy session for North region client acquisition
    - Case study on West region success for replication
    """
    
    print("Expert provides corrections and improvements...")
    
    feedback_id = system.incorporate_feedback(
        ai_analysis=ai_result['analysis'],
        expert_corrections=expert_corrections,
        dataset_summary=new_dataset_summary,
        tags=["revenue", "regional_analysis"],
        expert_id="analyst_jane"
    )
    
    print(f"✓ Incorporated expert feedback as annotation {feedback_id}")
    
    print("\n" + "=" * 70)
    print("PHASE 4: System Has Learned - Next Analysis Will Be Better")
    print("=" * 70)
    
    final_stats = system.get_system_stats()
    print(f"\nFinal System Stats:")
    print(json.dumps(final_stats, indent=2))
    
    print(f"""
    
    ✓ Learning Loop Complete!
    
    The system now has:
    - {final_stats['total_expert_annotations']} expert examples in knowledge base
    - Estimated accuracy: {final_stats['estimated_accuracy']:.1%}
    - Will continue improving with each expert annotation
    
    Next time a similar dataset arrives, the AI will:
    1. Retrieve these expert examples
    2. Generate analysis following their style
    3. Achieve higher accuracy due to more examples
    4. Require less expert correction over time
    """)


# ============================================================================
# ADDITIONAL HELPER FUNCTIONS
# ============================================================================

def simulate_active_learning_workflow():
    """
    Shows how to implement active learning:
    - AI identifies uncertain cases
    - Expert reviews only those cases
    - Dramatically reduces annotation workload
    """
    
    system = ExpertLearningSystem(anthropic_api_key="your-api-key-here")
    
    # Simulate multiple datasets to analyze
    datasets = [
        {"summary": "Dataset A", "details": "..."},
        {"summary": "Dataset B", "details": "..."},
        {"summary": "Dataset C", "details": "..."},
        # ... many more
    ]
    
    for dataset in datasets:
        # Generate AI analysis
        result = system.generate_ai_analysis(
            dataset_summary=dataset["summary"],
            dataset_details=dataset["details"]
        )
        
        # ACTIVE LEARNING: Only send to expert if confidence is low
        if result['confidence'] < 0.75:
            print(f"⚠️  Low confidence ({result['confidence']:.2%}) - sending to expert")
            # In real system, this would trigger expert review workflow
        else:
            print(f"✓ High confidence ({result['confidence']:.2%}) - auto-approved")
            # AI analysis used without expert review


if __name__ == "__main__":
    # Run the example workflow
    example_workflow()
    
    # Uncomment to see active learning example
    # simulate_active_learning_workflow()
