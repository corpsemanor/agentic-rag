"""
Agentic RAG Module - Crew.AI with Ollama via ChatOpenAI
Two agents: Researcher (validates facts) and Finisher (formats response)
"""
import logging
from crewai import LLM, Agent, Crew, Task
from app.config import GENERATION_MODEL, OLLAMA_HOST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_llm_instance(model: str) -> LLM:
    """Create LLM for CrewAI using CrewAI LLM pointing to Ollama"""
    return LLM(
        model=f"ollama/{model or GENERATION_MODEL}",
        base_url=OLLAMA_HOST
    )


def _create_researcher_agent(llm: LLM) -> Agent:
    """Agent that analyzes and synthesizes facts from retrieved documents"""
    return Agent(
        role="Information Analyst",
        goal="Synthesize and verify information from the provided context documents to directly answer the user's query. Focus only on the provided text.",
        backstory=(
            "You are a meticulous analyst. Your task is to examine the provided text snippets (context) "
            "and extract the key facts and information needed to answer a specific user query. "
            "You must stick strictly to the information present in the context. Do not invent, infer, "
            "or use outside knowledge. Your output is a consolidated summary of verified facts."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def _create_response_agent(llm: LLM) -> Agent:
    """Agent that formats the final response with citations"""
    return Agent(
        role="Senior Technical Writer",
        goal="Draft a clear, concise, and professional response to the user's query, integrating the validated facts and citing the sources correctly.",
        backstory=(
            "You are an expert communicator. You take a summary of validated facts and craft a final, "
            "user-facing response. The response must be polite, direct, and easy to understand. "
            "You must cite the relevant documents using the format [Source: X] where X is the source file name."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def _format_context_from_docs(docs: list) -> str:
    """Format documents for agent context with metadata"""
    if not docs:
        return "No documents available"

    formatted = []
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        score = doc.get("score", 0.0)
        doc_id = doc.get("document_id", doc.get("metadata", {}).get("id", i))
        source = doc.get("source", doc.get("metadata", {}).get("source_file", "unknown"))

        formatted.append(
            f"Document {i} [ID: {doc_id}, Source: {source}, Relevance: {score:.2f}]:\n{text}"
        )

    return "\n\n" + "=" * 50 + "\n\n".join(formatted)


async def run_rag_crew(model: str, user_query: str, retrieved_docs: list) -> dict:
    """Process query with CrewAI agents"""
    logger.info(f"[AGENT] Starting agent processing for query: {user_query[:50]}...")

    if not retrieved_docs:
        logger.warning("[AGENT] No documents retrieved")
        return {
            'response': 'No relevant information found.',
            'query': user_query,
            'sources': [],
            'count': 0
        }

    try:
        logger.info(f"[AGENT] Retrieved {len(retrieved_docs)} documents")
        logger.info("[AGENT] Step 1: Initializing LLM...")
        llm = _create_llm_instance(model)

        logger.info("[AGENT] Step 2: Formatting context from retrieved documents...")
        context = _format_context_from_docs(retrieved_docs)

        logger.info("[AGENT] Step 3: Creating agents...")
        researcher = _create_researcher_agent(llm)
        finisher = _create_response_agent(llm)

        logger.info("[AGENT] Step 4: Creating tasks...")
        research_task = Task(
            description=f"""
                User Query: "{user_query}"

                Context Documents:
                {context}

                Task:
                Review the context documents and extract all facts and information relevant to answering the user's query.
                Produce a summary of these facts. If no relevant information is found, state that clearly.
                """,
            agent=researcher,
            expected_output="A consolidated summary of facts derived *only* from the provided context, or a statement that no relevant information was found."
        )

        source_files = []
        seen_sources = set()
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get('source', doc.get('metadata', {}).get('source_file', 'unknown'))
            if source != 'unknown' and source not in seen_sources:
                source_files.append(f"[{i}] Source: {source}")
                seen_sources.add(source)
        sources_info = "\n".join(source_files) if source_files else "No sources available"

        finish_task = Task(
            description=f"""
                User Query: "{user_query}"

                Available Source Files:
                {sources_info}

                Task:
                Using the validated facts from the previous step, compose a final, comprehensive answer to the user's query.
                - Be clear, concise, and polite.
                - Cite your sources using the format [Source: {source_files[i] if i-1 < len(source_files) else 'unknown'}] based on the document numbers.
                - If the facts state that no information was found, inform the user politely.
                - At the very end of your response, list the primary source files you referenced.
                """,
            agent=finisher,
            expected_output="A final, well-formatted, user-facing answer that is fully supported by the provided context and includes citations.",
            context=[research_task]
        )

        logger.info("[AGENT] Step 5: Running Crew with Researcher and Finisher agents...")
        crew = Crew(
            agents=[researcher, finisher],
            tasks=[research_task, finish_task],
            verbose=True
        )

        logger.info("[AGENT] Researcher agent: Validating facts...")
        result = crew.kickoff()
        logger.info("[AGENT] Finisher agent: Formatting response with citations...")
        logger.info("[AGENT] Agent processing completed successfully")

        sources = [
            {
                'id': str(d.get('document_id', i)),
                'text': d['text'][:200],
                'score': d['score'],
                'source': d.get('source', 'unknown'),
                'metadata': d.get('metadata', {})
            }
            for i, d in enumerate(retrieved_docs, 1)
        ]

        return {
            'response': str(result),
            'query': user_query,
            'sources': sources,
            'count': len(sources)
        }

    except Exception as e:
        logger.error(f"[AGENT] Error during processing: {e}", exc_info=True)

        response_text = f"Based on retrieved information: {retrieved_docs[0]['text'][:300]}..."
        sources = [
            {'id': str(i), 'text': d['text'][:200], 'score': d['score']}
            for i, d in enumerate(retrieved_docs, 1)
        ]

        return {
            'response': response_text,
            'query': user_query,
            'sources': sources,
            'count': len(sources),
            'error': str(e)
        }