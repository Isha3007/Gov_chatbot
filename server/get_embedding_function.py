from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function():
   #  embeddings = BedrockEmbeddings(
   #     model_id="amazon.titan-embed-text-v2:0",
   #     credentials_profile_name="default",
   #     region_name="us-east-1"
   #  )
   # embeddings = OllamaEmbeddings(model="nomic-embed-text")
   model_name = "sentence-transformers/all-mpnet-base-v2"  # or all-MiniLM-L6-v2
   embeddings = HuggingFaceEmbeddings(model_name=model_name)
   return embeddings