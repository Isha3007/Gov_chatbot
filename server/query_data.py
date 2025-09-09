import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import OllamaLLM # optional
from get_embedding_function import get_embedding_function


load_dotenv()


CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma")


PROMPT_TEMPLATE = """
Answer the question based only on the following context:


{context}


---


Answer the question based on the above context: {question}
"""




def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Top-k retrieval
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Prepare prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # LLM choice (default: Gemini)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response = model.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # Extract distinct, normalized sources
    sources_set = set()
    for doc, _score in results:
        source_path = doc.metadata.get("source", "")
        if source_path.startswith("http"):
            # For websites, remove query parameters
            clean_source = source_path.split("?")[0]
        else:
            # For PDFs, just keep the filename
            clean_source = os.path.basename(source_path)
        sources_set.add(clean_source)

    sources = list(sources_set)
    return answer, sources
