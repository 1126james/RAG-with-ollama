from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1")
    return embeddings
