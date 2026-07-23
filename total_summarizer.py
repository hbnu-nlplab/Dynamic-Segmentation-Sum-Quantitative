import os
import openai
from dotenv import load_dotenv


class TotalSummarizer:
    def __init__(self, model="gpt-4o-mini", temperature=0.3, max_seg_summaries=None):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_seg_summaries = max_seg_summaries

    def _chat(self, messages):
        resp = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=30,
        )
        return resp.choices[0].message.content.strip()

    def generate_total_summary(self, segment_summaries):
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(segment_summaries, 1))

        prompt = f"""You are an expert in generating a final meeting summary from sub-topic-level summaries.
Below are summaries related to the overall meeting.
Your task is to write a final summary that represents the entire meeting.

Guidelines:
- Integrate the content naturally without redundancy.
- Write in a concise and clear style, similar to news articles.
- Write a single structurally organized summary that reflects the overall topic.
- The summary should be around 4~5 sentences.
- Output only the final summary. Do not include any additional explanation or titles.
- Write in Korean.

Segment Summaries:
{numbered}

Final Summary:""".strip()

        messages = [
            {"role": "system", "content": "당신은 회의 전체 요약을 작성하는 전문가입니다."},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages)

    def summarize(self, summary_map):
        """summary_map: {id: summary} (segment id 순서대로 정렬해 전체 요약 생성)"""
        ordered = [summary_map[sid].strip() for sid in sorted(summary_map) if summary_map[sid]]
        if not ordered:
            return ""

        if self.max_seg_summaries:
            ordered = ordered[:self.max_seg_summaries]

        return self.generate_total_summary(ordered)

    def generate_total_summary_with_topic(self, total_topic, sub_topic_summaries):
        combined = "\n".join(f"- {s}" for s in sub_topic_summaries)

        prompt = f"""You are an expert in generating a final meeting summary based on sub-topic-level summaries.

Your task is to write a single, coherent summary focusing on the total topic.

- total_topic:
"{total_topic}"

- sub_topic_summaries:
{combined}

Guidelines:
- Integrate the content naturally without redundancy.
- Write in a concise, news-style tone.
- Write around 4~5 sentences.
- Do NOT list sub-topics explicitly.
- Output only the summary.
- Write in Korean.""".strip()

        messages = [{"role": "user", "content": prompt}]
        return self._chat(messages)

    def summarize_with_topic(self, total_topic, topic_summary_map):
        """topic_summary_map: {topic: {"summary": str, ...}} (sub topic이 주어진 경우)"""
        summaries = [v["summary"].strip() for v in topic_summary_map.values() if v.get("summary")]
        if not summaries:
            return ""

        return self.generate_total_summary_with_topic(total_topic, summaries)
