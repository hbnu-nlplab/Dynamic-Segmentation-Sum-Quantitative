import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SubTopicSegmentMatcher:
    """이미 주어진 sub topic 목록에 대해, 자동 분할된 세그먼트 중 관련도 top-k를 매칭한다."""

    def __init__(self, segmenter, top_k=7, batch_size=4):
        self.segmenter = segmenter
        self.top_k = top_k
        self.batch_size = batch_size

    def _mean_pooled_embeddings(self, texts):
        tokenizer = self.segmenter.tokenizer
        model = self.segmenter.model
        embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                out = model(**inputs)
                h = out.last_hidden_state.to(torch.float32)
            mask = inputs["attention_mask"].unsqueeze(-1)
            mean = (h * mask).sum(1) / mask.sum(1)
            embs.append(mean.cpu().numpy())
        return np.vstack(embs)

    def _segment_embedding(self, sentences):
        return np.mean(self._mean_pooled_embeddings(sentences), axis=0, keepdims=True)

    def match(self, segments, sub_topics):
        """
        segments: [{"id": int, "sentences": [...]}, ...]
        sub_topics: [str, ...]
        -> {topic: [{"segment_id": int, "rank": int, "avg_similarity": float, "sentences": [...]}, ...]}
        """
        if not segments or not sub_topics:
            return {}

        if len(segments) <= self.top_k:
            return {
                topic: [
                    {
                        "segment_id": seg["id"],
                        "rank": i + 1,
                        "avg_similarity": 0.0,
                        "sentences": seg["sentences"],
                    }
                    for i, seg in enumerate(segments)
                ]
                for topic in sub_topics
            }

        segment_embeddings = np.vstack([self._segment_embedding(seg["sentences"]) for seg in segments])
        topic_embeddings = self._mean_pooled_embeddings(sub_topics)

        result = {}
        for topic, topic_emb in zip(sub_topics, topic_embeddings):
            similarities = cosine_similarity([topic_emb], segment_embeddings)[0]
            ranked = sorted(zip(segments, similarities), key=lambda pair: pair[1], reverse=True)[:self.top_k]
            result[topic] = [
                {
                    "segment_id": seg["id"],
                    "rank": i + 1,
                    "avg_similarity": float(sim),
                    "sentences": seg["sentences"],
                }
                for i, (seg, sim) in enumerate(ranked)
            ]
        return result
