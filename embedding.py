import logging
import os
import sys
from openai import OpenAI
from setting import API_KEY

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


logger = logging.getLogger(__name__)


class OPENAIClient(object):
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)

    def embed(self, texts, model="text-embedding-3-small"):
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=model)
        embeddings = [data.embedding for data in response.data]
        return embeddings


if __name__ == "__main__":
    openai_client = OPENAIClient()
    embedding = openai_client.embed(["jintian", "bge"])
    print(len(embedding), len(embedding[0]))
