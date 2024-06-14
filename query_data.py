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

def detect_language_and_intent(query_text: str):
    # Detect if the question is in English, Simplified Chinese, or Traditional Chinese
    # Also detect if the user requests a response in Chinese
    is_english = all('\u4e00' > char or char > '\u9fff' for char in query_text)
    request_chinese_response = "in chinese" in query_text.lower()

    if request_chinese_response:
        return "zh-sim"  # Default to Simplified Chinese for responses
    elif not is_english:
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

    # Detect the query language and intent
    language = detect_language_and_intent(query_text)
    if language == "zh-sim":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ZH_SIM)
    elif language == "zh-trad":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ZH_TRAD)
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EN)
    
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [(doc.metadata.get("id"), _score) for doc, _score in results]

    # Print the response from the LLM
    print(f"Response: {response_text}\n")

    # Debugging
    # Print the sources separately
    show_sources(sources)

    return response_text

def show_score(results):
    print('\n')
    for doc, score in results:
        print(f"Score: {score}, Doc ID: {doc.metadata['id']}")

def show_sources(sources):
    # Adjust column widths
    id_width = 50
    score_width = 10

    # Create a header for the table
    header = f"{'ID':<{id_width}}{'Score':>{score_width}}"
    separator = "-" * (id_width + score_width)

    # Format the sources to align ID and Score in columns
    formatted_sources = "\n".join([f"{id:<{id_width}}{score:>{score_width}.4f}" for id, score in sources])
    print(f"Sources:\n{header}\n{separator}\n{formatted_sources}")

if __name__ == "__main__":
    main()