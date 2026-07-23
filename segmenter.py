import logging

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("segmenter")


class MeetingSegmenter:
    def __init__(self, model_name, device_map="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device_map, torch_dtype="auto")
        self.model.eval()

        if self.tokenizer.sep_token is None or "<sep>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"sep_token": "<sep>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id

        print(f"✅ separator token: {self.sep_token} (id={self.sep_token_id})")

    def get_embeddings(self, sentences, max_length=32768):
        total_sent = len(sentences)
        chunks = []
        start_idx = 0

        while start_idx < total_sent:
            joined_text = f" {self.sep_token} ".join(sentences[start_idx:]) + f" {self.sep_token}"
            enc = self.tokenizer(
                joined_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).to(self.model.device)

            input_ids = enc["input_ids"][0].cpu().numpy()
            sep_positions = np.where(input_ids == self.sep_token_id)[0]
            covered = len(sep_positions)

            if covered == 0:
                print(f"⚠️ 문장이 max_length 초과 — 건너뜀 (idx={start_idx})")
                start_idx += 1
                continue

            try:
                with torch.no_grad():
                    hs = self.model(**enc).last_hidden_state.squeeze(0)
                    embs = hs[sep_positions].to(torch.float32).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                logger.error("Qwen 임베딩 계산 중 CUDA OOM 발생 → GPU 캐시를 비우고 예외를 전파합니다.")
                torch.cuda.empty_cache()
                raise

            chunk_end = start_idx + covered
            chunks.append((start_idx, chunk_end, embs))

            if total_sent > 100:
                print(f"📎 {chunk_end}/{total_sent} 문장 완료 (청크 {len(chunks)}개)")

            if chunk_end >= total_sent:
                break

            overlap = max(1, int(covered * 0.2))
            next_start = chunk_end - overlap
            start_idx = next_start if next_start > start_idx else chunk_end

        if not chunks:
            return np.array([])

        sent_candidates = [[] for _ in range(total_sent)]
        for ci, (cs, ce, embs) in enumerate(chunks):
            chunk_size = ce - cs
            for li, si in enumerate(range(cs, ce)):
                sent_candidates[si].append((ci, li, chunk_size, embs[li]))

        final_embeddings = []
        for si in range(total_sent):
            cands = sent_candidates[si]
            if len(cands) == 0:
                dim = chunks[0][2].shape[1]
                final_embeddings.append(np.zeros(dim, dtype=np.float32))
            elif len(cands) == 1:
                final_embeddings.append(cands[0][3])
            else:
                best_emb, best_sim = None, -np.inf
                for ci, li, chunk_size, emb in cands:
                    is_last = (ci == len(chunks) - 1)
                    _, _, chunk_embs = chunks[ci]
                    main_embs = chunk_embs if is_last else chunk_embs[:max(1, int(chunk_size * 0.8))]
                    center = np.mean(main_embs, axis=0, keepdims=True)
                    sim = cosine_similarity([emb], center)[0][0]
                    if sim > best_sim:
                        best_sim = sim
                        best_emb = emb
                final_embeddings.append(best_emb)

        embeddings = np.array(final_embeddings)
        return embeddings[:total_sent]

    def segment_sentences(self, sentences, embeddings):
        similarities = np.array([
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ])
        z_thresh = np.mean(similarities) - 1.5 * np.std(similarities)
        p_thresh = np.percentile(similarities, 5)
        threshold = (z_thresh + p_thresh) / 2
        change_points = np.where(similarities < threshold)[0] + 1

        segments, start = [], 0
        for cp in change_points:
            segments.append(sentences[start:cp])
            start = cp
        segments.append(sentences[start:])
        return segments, threshold, change_points

    def merge_small_segments(self, segments, embeddings, min_len=16):
        seg_ranges, idx = [], 0
        for seg in segments:
            seg_ranges.append((idx, idx + len(seg)))
            idx += len(seg)

        merged = segments.copy()
        ranges = seg_ranges.copy()

        i = 0
        while i < len(merged):
            if len(merged[i]) > min_len:
                i += 1
                continue

            cur_s, cur_e = ranges[i]
            cur_emb = np.mean(embeddings[cur_s:cur_e], axis=0, keepdims=True)
            sim_prev, sim_next = -1, -1

            if i > 0:
                ps, pe = ranges[i - 1]
                sim_prev = cosine_similarity(cur_emb, np.mean(embeddings[ps:pe], axis=0, keepdims=True))[0][0]
            if i < len(merged) - 1:
                ns, ne = ranges[i + 1]
                sim_next = cosine_similarity(cur_emb, np.mean(embeddings[ns:ne], axis=0, keepdims=True))[0][0]

            if sim_prev >= sim_next and i > 0:
                merged[i - 1].extend(merged[i])
                ranges[i - 1] = (ranges[i - 1][0], ranges[i][1])
                del merged[i], ranges[i]
                i -= 1
            elif i < len(merged) - 1:
                merged[i].extend(merged[i + 1])
                ranges[i] = (ranges[i][0], ranges[i + 1][1])
                del merged[i + 1], ranges[i + 1]
            else:
                i += 1

        return merged

    def segment(self, dialogue):
        """dialogue: [{"utterance": ..., "speaker": ...}, ...] -> [{"id": int, "sentences": [...]}, ...]"""
        sentences = [ut["utterance"] for ut in dialogue]

        embeddings = self.get_embeddings(sentences)
        segments, threshold, change_points = self.segment_sentences(sentences, embeddings)
        merged = self.merge_small_segments(segments, embeddings, min_len=16)

        return [{"id": i, "sentences": seg} for i, seg in enumerate(merged)]
