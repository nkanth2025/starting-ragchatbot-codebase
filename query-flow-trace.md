# Query Flow: Frontend to Backend

Here's the complete trace of a user query through the system:

---

## **Step 1: Frontend - User Submits Query**
`frontend/script.js:45-96`

```
User types question → clicks Send (or presses Enter)
         ↓
sendMessage() function triggered
         ↓
POST /api/query { query: "...", session_id: "..." }
```

The frontend:
1. Disables input while processing
2. Shows user's message in the chat
3. Displays a loading animation
4. Sends POST request with query and optional session ID

---

## **Step 2: FastAPI Endpoint**
`backend/app.py:56-74`

```python
@app.post("/api/query")
async def query_documents(request: QueryRequest):
    # Create session if not provided
    session_id = request.session_id or rag_system.session_manager.create_session()

    # Process query using RAG system
    answer, sources = rag_system.query(request.query, session_id)

    return QueryResponse(answer=answer, sources=sources, session_id=session_id)
```

---

## **Step 3: RAG System Orchestration**
`backend/rag_system.py:102-140`

```
rag_system.query(query, session_id)
         ↓
1. Get conversation history from SessionManager
2. Build prompt: "Answer this question about course materials: {query}"
3. Call AIGenerator with tools and history
4. Retrieve sources from ToolManager
5. Update conversation history
6. Return (response, sources)
```

---

## **Step 4: Claude API with Tool Use**
`backend/ai_generator.py:43-87`

```
AIGenerator.generate_response()
         ↓
Build API call:
  - System prompt (educational assistant instructions)
  - User message (the query)
  - Tools: [search_course_content]
  - Temperature: 0, max_tokens: 800
         ↓
Call Claude API → Response
         ↓
If stop_reason == "tool_use":
    → _handle_tool_execution()
Else:
    → Return direct text response
```

---

## **Step 5: Tool Execution (if Claude decides to search)**
`backend/ai_generator.py:89-135`

```
Claude returns tool_use request:
  { name: "search_course_content", input: { query: "...", course_name: "..." } }
         ↓
tool_manager.execute_tool("search_course_content", query=..., course_name=...)
         ↓
CourseSearchTool.execute() → formatted search results
         ↓
Send tool results back to Claude
         ↓
Claude generates final response incorporating search results
```

---

## **Step 6: Vector Search**
`backend/search_tools.py:52-86` → `backend/vector_store.py:61-100`

```
CourseSearchTool.execute(query, course_name?, lesson_number?)
         ↓
VectorStore.search()
         ↓
1. If course_name provided:
   → _resolve_course_name() - semantic search in course_catalog
   → Find best matching course title
         ↓
2. Build filter: { course_title, lesson_number }
         ↓
3. course_content.query(query_texts=[query], n_results=5, where=filter)
   → ChromaDB embeds query with SentenceTransformer
   → Finds top 5 semantically similar chunks
         ↓
4. Return SearchResults(documents, metadata, distances)
```

---

## **Step 7: Format Results & Return**
`backend/search_tools.py:88-114`

```
_format_results(SearchResults)
         ↓
For each result:
  "[Course Title - Lesson N]
   {chunk content}"
         ↓
Track sources: ["Course Title - Lesson N", ...]
         ↓
Return formatted string to Claude
```

---

## **Step 8: Response to Frontend**
`frontend/script.js:76-85`

```
Frontend receives:
{
  answer: "Claude's response...",
  sources: ["Course A - Lesson 1", "Course B - Lesson 3"],
  session_id: "abc123"
}
         ↓
Remove loading animation
Render answer with markdown (marked.js)
Show collapsible sources section
```

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND                                  │
│  User Input → sendMessage() → POST /api/query                       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI (app.py)                            │
│  query_documents() → rag_system.query()                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RAGSystem (rag_system.py)                     │
│  SessionManager → AIGenerator → ToolManager                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AIGenerator (ai_generator.py)                  │
│  Claude API Call → Tool Use Decision                                │
│       ↓                                                             │
│  If tool_use: execute tool → send results → get final response      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ (if searching)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   CourseSearchTool (search_tools.py)                │
│  execute() → VectorStore.search()                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     VectorStore (vector_store.py)                   │
│  ChromaDB: embed query → semantic search → return top 5 chunks      │
└─────────────────────────────────────────────────────────────────────┘
```
