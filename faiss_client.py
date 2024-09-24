import json
import logging
import os
import faiss
import numpy as np
from embedding import OPENAIClient
from tqdm import tqdm
from rerank import SimilarityRanker

logger = logging.getLogger(__name__)


def dict_to_string(data_dict):
    # 用于存储拼接的字符串
    result_str = ""

    # 遍历字典，将键和值拼接为字符串
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # 如果值是字典，递归拼接
            value_str = dict_to_string(value)
            result_str += f"{key}: {value_str}\n"
        elif isinstance(value, list):
            # 如果值是列表，拼接列表中的每个元素
            value_str = ", ".join(str(item) for item in value)
            result_str += f"{key}: [{value_str}]\n"
        else:
            # 否则直接拼接
            result_str += f"{key}: {value}\n"

    return result_str.strip()  # 去除最后一个多余的换行符


class FaissClient(object):
    def __init__(self):
        self.index = faiss.IndexFlatL2(1536)
        self.embedding_client = OPENAIClient()
        self.bge_rerank = SimilarityRanker()
        self.version = "v2"
        self.rate = 2
        self.texts = []
        self.texts_process = []

    def build_index(self, texts_process: list[str], texts: list[str], num_batches: int = 100):
        batch_size = len(texts_process) // num_batches  # 计算每批的大小
        start_idx = 0

        for _ in tqdm(range(num_batches), total=num_batches):
            end_idx = start_idx + batch_size if start_idx + batch_size <= len(texts_process) else len(texts_process)
            batch_texts_process = texts_process[start_idx:end_idx]
            batch_texts = texts[start_idx:end_idx]  # 对应的完整文本
            vectors = np.array(self.embedding_client.embed(batch_texts_process)).astype(np.float32)
            self.index.add(vectors)

            # 存储文本及其对应的索引
            self.texts.extend(batch_texts)
            self.texts_process.extend(batch_texts_process)  # 添加此行以保存处理后的文本块
            start_idx = end_idx

    def save_index(self, index_file_path, texts_file_path, texts_process_file_path):
        """Save the FAISS index and the texts with their processed versions."""
        faiss.write_index(self.index, index_file_path)
        with open(texts_file_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=4)
        with open(texts_process_file_path, "w", encoding="utf-8") as f:
            json.dump(self.texts_process, f, ensure_ascii=False, indent=4)

    def load_index(self, index_file_path, texts_file_path, texts_process_file_path):
        """Load the FAISS index and the texts with their processed versions."""
        self.index = faiss.read_index(index_file_path)
        with open(texts_file_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(texts_process_file_path, "r", encoding="utf-8") as f:
            self.texts_process = json.load(f)

    def load_texts_from_folder(self, folder_path):
        texts = []
        texts_process = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                    for item in data:
                        full_text = dict_to_string(item)
                        texts.append(full_text)
                        code_part = filename.split(".")[0]
                        if "Description" in item:
                            description = item["Description"]
                            if isinstance(description, list):
                                description_str = ", ".join(map(str, description))
                            else:
                                description_str = str(description)
                            texts_process.append(f"{code_part}: {description_str}")
                        else:
                            texts_process.append(f"{code_part}: {full_text}")
        return texts, texts_process

    def data_process(self, folder_path):
        texts = []
        texts_process = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                if filename in ["networkx.json", "littleballoffur.json"]:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            if "Description" in item:
                                full_text = dict_to_string(item)
                                texts.append(full_text)
                                code_part = filename.split(".")[0]
                                description = item["Description"]
                                if isinstance(description, list):
                                    description_str = ", ".join(map(str, description))
                                else:
                                    description_str = str(description)
                                texts_process.append(f"{code_part}: {description_str}")
                elif filename in ["igraph.json", "igraph,json", "graspologic.json", "cdlib.json"]:
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data:
                            if "Section_id" in item and "Description" in item:
                                description = item["Description"]
                            if "Section_id" in item and "Description" not in item:
                                for key, value in item.items():
                                    if "Description" in value:
                                        code_part = filename.split(".")[0]
                                        description_str = description + ". " + value.get("Description")
                                        texts_process.append(f"{code_part}: {description_str}")
                                        full_text = dict_to_string(item)
                                        texts.append(full_text)

        return texts, texts_process

    def search_rerank(self, query: str, topk=5):
        self.load_index(
            f"index/{self.version}/faiss_index.bin",
            f"index/{self.version}/faiss_texts.json",
            f"index/{self.version}/faiss_texts_process.json",
        )
        query_vector = np.array(self.embedding_client.embed([query])).astype(np.float32)
        _, indices = self.index.search(query_vector, self.rate * topk)
        results = [self.texts_process[i] for i in indices[0]]
        reranked_results, reranked_scores = self.bge_rerank.get_top_k_similar(query, results, top_k=topk)
        reranked_results_full_text = [self.texts[self.texts_process.index(result)] for result in reranked_results]
        return reranked_scores, "\n".join(reranked_results_full_text)

    def search(self, query: str, topk=5):
        self.load_index(
            f"index/{self.version}/faiss_index.bin",
            f"index/{self.version}/faiss_texts.json",
            f"index/{self.version}/faiss_texts_process.json",
        )
        query_vector = np.array(self.embedding_client.embed([query])).astype(np.float32)
        _, indices = self.index.search(query_vector, topk)
        results_full_text = [self.texts[i] for i in indices[0]]
        return None, "\n".join(results_full_text)


# Usage example
if __name__ == "__main__":
    faiss_index = FaissClient()
    # folder_path = "database/GraphPro-master/doc datasets"  # Replace with the folder path containing JSON files
    # texts, texts_process = faiss_index.data_process(folder_path)
    # faiss_index.build_index(texts_process, texts)
    # faiss_index.save_index("v2/index/faiss_index.bin", "v2/index/faiss_texts.json", "v2/index/faiss_texts_process.json")
    distances, results = faiss_index.search(
        query="As a Risk Analyst, you have been tasked with assessing potential risks in a social network analysis project involving the Les Mis√©rables characters. You are required to utilize the Louvain method available in the cdlib library to perform community detection on the Les Mis√©rables graph, which can be loaded from a GML file named lesmis.gml. Additionally, you need to calculate the Erdos-Renyi modularity of the detected communities. This modularity score is essential in understanding the strength of community structures within the network.",
        topk=3,
    )
    print(distances)
    print(results)
