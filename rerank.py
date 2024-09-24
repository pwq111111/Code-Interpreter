from FlagEmbedding import FlagReranker


class SimilarityRanker:
    def __init__(self, model_name="F:/models/bge-reranker-v2-m3", use_fp16=True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def get_top_k_similar(self, query, candidates, top_k=3):
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.reranker.compute_score(pairs)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_k_candidates = [candidates[i] for i in top_k_indices]
        top_k_scores = [scores[i] for i in top_k_indices]
        return top_k_candidates, top_k_scores
