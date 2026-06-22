## ADDED Requirements

### Requirement: Knowledge Base Construction
The system SHALL build a structured knowledge base from inline LangChain documentation covering framework overview, component descriptions, best practices, and FAQ content. The knowledge base SHALL support both dense vector retrieval (via ChromaDB + HuggingFace embeddings) and sparse keyword retrieval (via BM25).

#### Scenario: Build knowledge base from inline docs
- **WHEN** the script initializes
- **THEN** it SHALL create Document chunks from inline documentation text
- **AND** build both a ChromaDB vector index and a BM25 keyword index
- **AND** support querying with top_k parameter

### Requirement: Deep Agent Creation with RAG Tools
The system SHALL use `deepagents.create_deep_agent()` to create an autonomous RAG Agent equipped with at least the following tools: `vector_search`, `keyword_search`, `hybrid_search`, `evaluate_relevance`. The Agent SHALL be configured with `ollama:llama3.2:1b` as the default model.

#### Scenario: Create RAG Agent
- **WHEN** `create_deep_agent()` is called with RAG tools
- **THEN** the Agent SHALL be able to invoke those tools autonomously during conversation
- **AND** SHALL handle tool call errors gracefully with fallback messages

### Requirement: Query Decomposition
The Agent SHALL decompose complex multi-part questions into sub-queries using `write_todos` for task planning, then retrieve information for each sub-query separately.

#### Scenario: Decompose multi-part question
- **WHEN** the user asks a question with multiple sub-topics (e.g. "What is LangChain and how does its agent system work?")
- **THEN** the Agent SHALL create a todo list with sub-tasks for each sub-topic
- **AND** SHALL retrieve relevant documents for each sub-query

### Requirement: Multi-Retriever Routing
The Agent SHALL support routing queries to different retriever strategies: dense vector search for semantic matching, sparse keyword search for exact term matching, and hybrid search combining both. The Agent SHALL decide which strategy to use based on the query characteristics.

#### Scenario: Route to correct retriever
- **WHEN** the user asks a conceptual question (e.g. "explain the architecture")
- **THEN** the Agent SHALL prefer vector search for semantic understanding
- **WHEN** the user asks about specific named terms (e.g. "RunnablePassthrough")
- **THEN** the Agent SHALL use hybrid search that includes keyword matching

### Requirement: Result Evaluation and Iteration
The Agent SHALL evaluate retrieved results for relevance and completeness. If results are insufficient, the Agent SHALL attempt to re-retrieve with improved queries, up to a maximum of 3 iterations.

#### Scenario: Evaluate and re-retrieve
- **WHEN** retrieved results do not contain sufficient information to answer the question
- **THEN** the Agent SHALL generate improved search queries
- **AND** SHALL perform additional retrieval cycles
- **AND** SHALL stop after reaching the maximum iteration limit (3)

### Requirement: Multi-Source Information Fusion
The Agent SHALL combine information from multiple retrieval sources (vector + keyword + hybrid) into a coherent synthesis, resolving duplicate content and prioritizing the most relevant passages.

#### Scenario: Fusion from multiple retrievers
- **WHEN** the Agent uses multiple retriever strategies for the same question
- **THEN** it SHALL merge and deduplicate results from all strategies
- **AND** SHALL prioritize results based on relevance scores

### Requirement: Source Citation and Provenance
The Agent SHALL include source citations in its final answer, referencing the original document chunks. Each citation SHALL include the document title and a relevance indicator.

#### Scenario: Answer with citations
- **WHEN** the Agent generates a final answer
- **THEN** the answer SHALL reference the source documents with identifiers (e.g., [Doc1], [Doc2])
- **AND** SHALL list the referenced sources at the end of the answer

### Requirement: Human-in-the-Loop Interrupt
The Agent SHALL support human-in-the-loop approval for critical operations (file writes) via `interrupt_on`, with `MemorySaver` checkpointer, allowing a human to review and approve or reject the Agent's action before execution.

#### Scenario: Interrupt before file write
- **WHEN** the Agent attempts to write a file (e.g., saving retrieval results)
- **THEN** execution SHALL pause before the write operation
- **AND** SHALL wait for human approval or rejection via resume mechanism
