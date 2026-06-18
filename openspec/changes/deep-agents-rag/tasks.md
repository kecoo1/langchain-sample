## 1. File Setup and Dependencies

- [ ] 1.1 Create `06_advanced_topics/38_deep_agents_rag.py` with docstring header, imports (deepagents, langchain-ollama, chromadb, etc.), and try-catch for deepagents availability
- [ ] 1.2 Implement `_check_available()` and model/configuration constants

## 2. Knowledge Base Construction

- [ ] 2.1 Define structured `KNOWLEDGE_BASE` dict with LangChain FAQ categories (Basics, Components, Best Practices, Troubleshooting, FAQ)
- [ ] 2.2 Implement `build_knowledge_base()`: convert dict to Document list, apply RecursiveCharacterTextSplitter, build ChromaDB vectorstore + BM25 keyword index
- [ ] 2.3 Implement `vector_search(query, top_k)`, `keyword_search(query, top_k)`, `hybrid_search(query, top_k)` as @tool functions

## 3. RAG Evaluation Tool

- [ ] 3.1 Implement `evaluate_relevance(query, results_str)` as @tool: checks if retrieved content can answer the query, returns JSON with `sufficient` flag and `reason` field
- [ ] 3.2 Add result deduplication helper and relevance scoring utility

## 4. Deep Agent Examples (6 examples)

- [ ] 4.1 **示例 1: Basic RAG Agent** — Create deep_agent with RAG tools, invoke with single question, show tool calling trace
- [ ] 4.2 **示例 2: Query Decomposition** — Multi-part question triggers write_todos, Agent splits and retrieves per sub-query
- [ ] 4.3 **示例 3: Multi-Retriever Routing** — Agent chooses between vector/keyword/hybrid based on question type, shows decision process
- [ ] 4.4 **示例 4: Result Evaluation & Iteration** — Agent evaluates results, re-retrieves if insufficient, up to 3 iterations
- [ ] 4.5 **示例 5: Multi-Source Fusion with Citations** — Agent combines results from all retrievers, deduplicates, answers with [DocN] citations
- [ ] 4.6 **示例 6: Human-in-the-Loop with Virtual Filesystem** — interrupt_on for file writes, agent saves retrieval report to virtual filesystem, pauses for approval

## 5. Teaching Notes and Finalization

- [ ] 5.1 Add teaching notes for each example (what's happening, why it matters, model limitations)
- [ ] 5.2 Add appendix with model capability assessment specific to Agent RAG
- [ ] 5.3 Add `if __name__ == "__main__"` guard with example on/off switches
- [ ] 5.4 Run python syntax check: `python -c "import ast; ast.parse(open('06_advanced_topics/38_deep_agents_rag.py').read())"`
