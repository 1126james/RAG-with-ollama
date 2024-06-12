from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = 'chroma'

def get_chunk_content_by_id(chunk_id, chroma_db):
    """
    Retrieves the content of a document chunk based on its ID from a Chroma DB.

    :param chunk_id: The ID of the chunk to retrieve, formatted as "filename:page:chunk".
    :param chroma_db: The Chroma DB instance to query.
    :return: The content of the chunk if found, otherwise returns None.
    """
    # Retrieve the document by ID
    results = chroma_db.get(ids=[chunk_id])
    if results and results["documents"]:
        # Extract the index of the document based on its ID in the 'ids' list
        try:
            index = results["ids"].index(chunk_id)
            # Assuming the actual content is stored under 'data' key
            return results["documents"][index]
        except (ValueError, IndexError):
            print(f"No document found with ID: {chunk_id}")
            return None
    else:
        print(f"No document found with ID: {chunk_id}")
        return None

# Example of using the Chroma DB
def main():
    # Set up the Chroma DB instance (assuming it's already populated and configured)
    chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())  # Adjust as needed
    while True:
        # Example chunk ID
        chunk_id = str(input("\n\nExample: ticket_to_ride.pdf:3:5\n👉 Chunk_id: "))
        content = get_chunk_content_by_id(chunk_id, chroma_db)
        if content:
            print(f"Content for ID '{chunk_id}':\n\n {content}")
        else:
            print("Content not found.")

if __name__ == "__main__":
    main()