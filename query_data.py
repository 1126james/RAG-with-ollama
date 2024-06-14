import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from prompt_templates import PROMPT_TEMPLATE_EN, PROMPT_TEMPLATE_ZH_SIM, PROMPT_TEMPLATE_ZH_TRAD

CHROMA_PATH = "chroma"

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def detect_language(query_text: str) -> str:
    # Simple heuristic to detect if the question is in English, Simplified Chinese, or Traditional Chinese
    if any('\u4e00' <= char <= '\u9fff' for char in query_text):
        if any('\u3400' <= char <= '\u4DBF' or '\u4E00' <= char <= '\u9FFF' for char in query_text):
            return "zh-sim"
        else:
            return "zh-trad"
    return "en"

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=6)

    # Ensure results are sorted by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Debugging
    # show_score(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Detect the query language
    language = detect_language(query_text)
    if language == "zh-sim":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ZH_SIM)
    elif language == "zh-trad":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ZH_TRAD)
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EN)
    
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text

def show_score(results):
    print('\n')
    for doc, score in results:
        print(f"Score: {score}, Doc ID: {doc.metadata['id']}")

if __name__ == "__main__":
    main()