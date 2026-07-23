import os
import logging
import threading
from collections import defaultdict

from segmenter import MeetingSegmenter
from topic_matcher import SubTopicSegmentMatcher
from seg_summarizer import SegmentSummarizer
from seg_topic_generator import SegTopicGenerator
from sub_summarizer import SubTopicSummarizer
from total_summarizer import TotalSummarizer
from speaker_summarizer import SpeakerSummarizer
from kormo_backend import KormoBackend

logger = logging.getLogger("pipeline")

TOP_K = 7


def _speaker_labeled_block(sentences, speaker_map):
    return "\n".join(f"{speaker_map.get(s, 'UNKNOWN')}: {s}" for s in sentences)


def _speaker_map(dialogue):
    return {
        ut["utterance"]: ut["speaker"]
        for ut in dialogue
        if "utterance" in ut and "speaker" in ut
    }


class SummaryPipeline:
    def __init__(self, model_name, openai_model, kormo_model_dir):
        # Qwen(세그멘테이션)과 Kormo(생성)는 동시에 GPU 메모리를 쓸 수 있어 서로 다른 GPU에
        # 배치할 수 있도록 각각 device_map을 환경변수로 지정 가능하게 둔다 (기본은 둘 다 auto)
        qwen_device_map = os.getenv("QWEN_DEVICE_MAP", "auto")
        kormo_device_map = os.getenv("KORMO_DEVICE_MAP", "auto")

        print("Qwen 임베딩 모델 로딩 중...")
        self.segmenter = MeetingSegmenter(model_name, device_map=qwen_device_map)
        print("Qwen 임베딩 모델 로딩 완료")

        self.topic_matcher = SubTopicSegmentMatcher(self.segmenter, top_k=TOP_K)

        self.seg_summarizer = SegmentSummarizer(openai_model)
        self.topic_generator = SegTopicGenerator(openai_model)
        self.sub_summarizer = SubTopicSummarizer(openai_model)
        self.total_summarizer = TotalSummarizer(openai_model)
        self.speaker_summarizer = SpeakerSummarizer(openai_model)

        print("Kormo 모델 로딩 중...")
        self.kormo_backend = KormoBackend(kormo_model_dir, device_map=kormo_device_map)
        self.kormo_backend.load()
        print("Kormo 모델 로딩 완료")

        # GPU에 올라간 모델(Qwen 세그멘테이션 / Kormo 생성)은 한 번에 한 요청만 사용하도록 보호
        self._gpu_lock = threading.Lock()

    def run(self, filename, dialogue, pipeline_type, sub_topics=None, total_topic="", model="gpt"):
        if pipeline_type == "w_st":
            return self._run_with_subtopic(filename, dialogue, sub_topics, total_topic, model)
        return self._run_without_subtopic(filename, dialogue, model)

    # ------------------------------------------------------------------
    # wo_st
    # ------------------------------------------------------------------

    def _run_without_subtopic(self, filename, dialogue, model):
        with self._gpu_lock:
            segments = self.segmenter.segment(dialogue)

        speaker_map = _speaker_map(dialogue)

        try:
            return self._build_wo_st(filename, segments, speaker_map, use_kormo=(model == "kormo"))
        except Exception as e:
            if model == "kormo":
                raise
            logger.error(
                "GPT로 wo_st 파이프라인 실행 중 오류 발생 → Kormo로 처음부터 다시 시도합니다: %s",
                e, exc_info=True,
            )
            return self._build_wo_st(filename, segments, speaker_map, use_kormo=True)

    def _build_wo_st(self, filename, segments, speaker_map, use_kormo):
        def gen_segsum(sentences, span_block):
            if use_kormo:
                with self._gpu_lock:
                    return self.kormo_backend.generate_segsum(span_block)
            return self.seg_summarizer.summarize_segment(sentences)

        def gen_topic(sentences, span_block, summary):
            if use_kormo:
                with self._gpu_lock:
                    return self.kormo_backend.generate_subtopic(span_block, summary)
            return self.topic_generator.generate_topic(sentences, summary)

        def gen_speaker(sid, speaker, utts, sentences, span_block, utt_block):
            if use_kormo:
                with self._gpu_lock:
                    return self.kormo_backend.generate_speakersum(span_block, speaker, utt_block)
            return self.speaker_summarizer.summarize_speaker(sid, speaker, utts, sentences)

        def gen_total(ordered):
            if use_kormo:
                with self._gpu_lock:
                    return self.kormo_backend.generate_totalsum(ordered)
            return self.total_summarizer.generate_total_summary(ordered)

        summary_map = {}
        for seg in segments:
            sentences = seg["sentences"]
            span_block = _speaker_labeled_block(sentences, speaker_map)
            summary = gen_segsum(sentences, span_block)
            if summary:
                summary_map[seg["id"]] = summary

        topic_map = {}
        for seg in segments:
            sid = seg["id"]
            sentences = seg["sentences"]
            summary = summary_map.get(sid, "")
            span_block = _speaker_labeled_block(sentences, speaker_map)
            topic = gen_topic(sentences, span_block, summary)
            if topic and not topic.startswith("[ERROR]"):
                topic_map[sid] = topic

        speaker_result = {}
        for seg in segments:
            sid = seg["id"]
            sentences = seg["sentences"]
            span_block = _speaker_labeled_block(sentences, speaker_map)

            speaker2utts = defaultdict(list)
            for sent in sentences:
                speaker2utts[speaker_map.get(sent, "UNKNOWN")].append(sent)

            speaker_summaries = {}
            for speaker, utts in speaker2utts.items():
                utt_block = "\n".join(utts)
                summary = gen_speaker(sid, speaker, utts, sentences, span_block, utt_block)
                speaker_summaries[speaker] = summary
            speaker_result[sid] = speaker_summaries

        ordered = [summary_map[sid] for sid in sorted(summary_map) if summary_map[sid]]
        total_summary = gen_total(ordered) if ordered else ""

        merged_segments = [
            {
                "id": seg["id"],
                "summary": summary_map.get(seg["id"], ""),
                "topic": topic_map.get(seg["id"], ""),
                "speaker_summaries": speaker_result.get(seg["id"], {}),
            }
            for seg in segments
        ]

        return {
            "file": filename,
            "type": "wo_st",
            "model": "kormo" if use_kormo else "gpt",
            "total_summary": total_summary,
            "segments": merged_segments,
        }

    # ------------------------------------------------------------------
    # w_st
    # ------------------------------------------------------------------

    def _run_with_subtopic(self, filename, dialogue, sub_topics, total_topic, model):
        with self._gpu_lock:
            segments = self.segmenter.segment(dialogue)
            topic_segment_map = self.topic_matcher.match(segments, sub_topics)

        speaker_map = _speaker_map(dialogue)

        try:
            return self._build_w_st(
                filename, sub_topics, topic_segment_map, speaker_map, total_topic, use_kormo=(model == "kormo")
            )
        except Exception as e:
            if model == "kormo":
                raise
            logger.error(
                "GPT로 w_st 파이프라인 실행 중 오류 발생 → Kormo로 처음부터 다시 시도합니다: %s",
                e, exc_info=True,
            )
            return self._build_w_st(
                filename, sub_topics, topic_segment_map, speaker_map, total_topic, use_kormo=True
            )

    def _build_w_st(self, filename, sub_topics, topic_segment_map, speaker_map, total_topic, use_kormo):
        def gen_topic_summary(topic, sentences, span_block):
            if use_kormo:
                # w_st 전용 학습 태스크가 아직 없어, Kormo는 wo_st의 segsum 프롬프트를 그대로 재사용한다
                with self._gpu_lock:
                    return self.kormo_backend.generate_segsum(span_block)
            return self.sub_summarizer.summarize_topic(topic, sentences)

        def gen_speaker(topic, speaker, utts, all_items, span_block, utt_block):
            if use_kormo:
                with self._gpu_lock:
                    return self.kormo_backend.generate_speakersum(span_block, speaker, utt_block)
            return self.speaker_summarizer.summarize_speaker_for_topic(topic, speaker, utts, all_items)

        def gen_total(ordered_summaries):
            if use_kormo:
                # Kormo의 totalsum 태스크는 topic 조건화 없이 학습되어 total_topic은 무시하고 재사용한다
                with self._gpu_lock:
                    return self.kormo_backend.generate_totalsum(ordered_summaries)
            return self.total_summarizer.generate_total_summary_with_topic(total_topic, ordered_summaries)

        summary_map = {}
        for topic, segs in topic_segment_map.items():
            sorted_segs = sorted(segs, key=lambda x: x.get("rank", 999))
            sentences = [s for seg in sorted_segs for s in seg.get("sentences", [])]
            if not sentences:
                continue

            span_block = _speaker_labeled_block(sentences, speaker_map)
            summary = gen_topic_summary(topic, sentences, span_block)
            summary_map[topic] = {"summary": summary, "sentence_count": len(sentences)}

        speaker_result = {}
        for topic, segs in topic_segment_map.items():
            sorted_segs = sorted(segs, key=lambda x: x.get("rank", 999))
            all_sentences = [s for seg in sorted_segs for s in seg.get("sentences", [])]
            span_block = _speaker_labeled_block(all_sentences, speaker_map)

            speaker2utts = defaultdict(list)
            all_items = []
            for sent in all_sentences:
                spk = speaker_map.get(sent, "UNKNOWN")
                speaker2utts[spk].append(sent)
                all_items.append((spk, sent))

            speaker_summaries = {}
            for speaker, utts in speaker2utts.items():
                utt_block = "\n".join(utts)
                summary = gen_speaker(topic, speaker, utts, all_items, span_block, utt_block)
                speaker_summaries[speaker] = summary
            speaker_result[topic] = speaker_summaries

        ordered_summaries = [v["summary"].strip() for v in summary_map.values() if v.get("summary")]
        total_summary = gen_total(ordered_summaries) if ordered_summaries else ""

        topics_out = []
        for topic in sub_topics:
            if topic not in summary_map:
                continue
            topics_out.append({
                "topic": topic,
                "summary": summary_map[topic]["summary"],
                "speaker_summaries": speaker_result.get(topic, {}),
            })

        return {
            "file": filename,
            "type": "w_st",
            "model": "kormo" if use_kormo else "gpt",
            "total_topic": total_topic,
            "total_summary": total_summary,
            "topics": topics_out,
        }
