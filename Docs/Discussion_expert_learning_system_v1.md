# Discussion — Expert Learning System V1

Starting from the question “let’s talk about the embeddings. i understand it is a transformation function,” this document records the design notes and planned improvements that emerged from the conversation with Codex CLI (gpt-5.2-codex) on 02/01/2026.

## Embedding & Retrieval Flow
1. **Dataset fields**: `dataset_details` is stored verbatim for prompt context; `expert_analysis` is the field that carries the embedding. `Dataset_summary` is a short textual description used when searching for similar cases.
2. **Embedding service**: Gemini’s `text-embedding-004` via the custom `GeminiGenaiEmbedding` class populates LanceDB’s vector column, yielding 768-dimensional float vectors. Retrieval happens with `table.search(dataset_summary)` using the new dataset’s summary and the stored expert vectors, then similar analyses feed `_build_analysis_prompt()` for generation.
3. **Generation**: Gemini `gemini-2.0-flash` receives the augmented prompt (retrieved examples + new data + instructions) and answers in natural language. The human expert then reviews this draft, edits it if necessary, and stores the corrected version back into the database (with `incorporate_feedback()` tagging the entry as `ai_corrected`).

## Patterns
- Patterns come from human experts. Each annotation’s `patterns_found` list is a structured JSON-like list defined and filled before hitting `store_expert_annotation()`. The AI draft mentions patterns narratively but does not output the structured objects.
- *Improvement idea*: extend prompts so Gemini returns a JSON array of patterns (type/location/severity), validate/parse it, then let the expert confirm before persisting, giving the system semi-structured pattern data with less manual transcription.

## Structured Summaries
- Currently `dataset_summary` is a single string; it should stay concise for similarity search.
- *Improvement idea*: augment it with JSON metadata (e.g., period/metric/region) stored alongside the text. This allows filtering/tagging or more targeted prompts while keeping the semantic summary for embeddings.

## Embedding Source
- The implementation uses Gemini for embeddings, but the pipeline doesn’t depend on Gemini specifically. A locally hosted embedding model (e.g., `sentence-transformers`, Ollama, or another open-source encoder) could replace the `GeminiGenaiEmbedding` class by registering a new `TextEmbeddingFunction` with LanceDB.
- *Improvement idea*: evaluate pros/cons (latency, accuracy, cost) and eventually switch to local embeddings if it fits the use case.

## Database Workflow
- Right now the example script rebuilds three sample annotations and demonstrates a workflow, which is useful for learning but not for production.
- *Improvement idea*: refactor the script so it reads/writes dynamically against the existing `expert_learning_system_v1_db/`, exposing reusable functions (CLI/interactive prompts) that append real annotations, retrieve similar examples, run generation, and process feedback without replaying canned data each run.

## Summary of Planned Improvements
1. Ask Gemini for structured patterns to reduce manual transcription and better feed downstream analytics.
2. Store structured `dataset_summary` metadata (JSON) alongside the textual summary for richer filtering and prompting.
3. Research and possibly adopt local embeddings so the vector store stops depending on Gemini's embedding API.
4. Refactor the script to work dynamically on the persistent database instead of running the same simulated demo on each execution.

---

## 02/02/2026 — Discussion with Claude Opus 4.5

This session focused on planning V2 of the expert learning system. Decision: keep V1 as-is (a working demo/reference) and build V2 as the functional tool.

### V1 Status
- V1 remains a demonstration script showing the RAG workflow
- Useful for learning and reference
- Not intended for production use

### V2 Planning: CLI-First Approach

**Sequencing decision:** Build the CLI first with hardcoded values, then extract configuration later.

Rationale:
- Get something functional faster
- No multiple use cases yet (config extraction solves a future problem)
- Building the CLI reveals what actually needs to be configurable
- Easier to extract config from working code than to design speculatively

**Proposed CLI commands:**
```
python cli.py add        # Add new annotation
python cli.py analyze    # Analyze new dataset with AI
python cli.py list       # Show all annotations
python cli.py search     # Find similar past analyses
```

### V2 Planning: Configuration System (Later)

After CLI works, extract settings to config files per use case:
- Database path and name
- Pattern vocabulary (anomaly, trend, seasonality, etc.)
- Tag vocabulary
- AI prompt templates
- Output format expectations

Structure:
```
projects/
├── revenue_analysis/
│   ├── config.json
│   └── database/
├── support_tickets/
│   ├── config.json
│   └── database/
```

### V2 Planning: Simplified Example Data

Current V1 has three diverse annotations (revenue, churn, website traffic). For V2 development, simplify to a single domain: **revenue analysis only**.

Benefits:
- Retrieval finds genuinely similar cases (better signal)
- Consistent pattern vocabulary
- Easier to debug and reason about
- Clear test: Q1 2024 data should match Q1-Q4 2023 data

Proposed revenue-only annotations:
| # | Dataset | Key Finding |
|---|---------|-------------|
| 1 | Q1-Q4 2023 by region | Q3 anomaly, West strong, seasonal |
| 2 | Q1-Q4 2022 by region | Steady growth, East underperformed |
| 3 | Q1-Q4 2023 by product | Product A declining, C growing |

### V2 Planning: Simulated Expert Corpus (~50 cases)

Idea: Create a larger dataset of simulated expert annotations to:
1. Serve as ground truth / reference
2. Gradually feed cases to the AI
3. Measure how AI improves as database grows
4. Test if AI can replicate expert analysis style

**Scenario types for variety:**
| Type | Count | Description |
|------|-------|-------------|
| Normal/stable | 10 | No major issues |
| Single anomaly | 10 | One quarter/region off |
| Seasonal pattern | 8 | Predictable peaks |
| Growth trend | 6 | Upward movement |
| Decline trend | 6 | Downward movement |
| Regional variance | 5 | One region different |
| Complex/multi-pattern | 5 | Multiple signals |

**Generation approach:** Template + variation with manual review.

**Metrics to track:**
- Pattern detection rate (did AI find same patterns?)
- Recommendation quality
- Style matching
- Improvement curve as DB grows

### Next Steps for V2
1. Design the 50-case simulated corpus structure
2. Build CLI with basic commands (add, analyze, list)
3. Test with revenue-only data
4. Add configuration extraction when needed
5. Expand to other domains once revenue works well
