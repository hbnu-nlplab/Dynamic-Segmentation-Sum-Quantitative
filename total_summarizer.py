import os
import openai
from dotenv import load_dotenv


class TotalSummarizer:
    def __init__(self, model="gpt-4o-mini", temperature=0.3):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature

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

    def generate_total_topic(self, segment_summaries, segment_topics):
        numbered_summaries = "\n".join(f"{i}. {s}" for i, s in enumerate(segment_summaries, 1))
        topics_list = "\n".join(f"- {t}" for t in segment_topics)

        prompt = f"""You are an expert at expressing the overall topic of a meeting as a Korean noun phrase.

Below are the segment-level summaries and segment-level topics from one meeting.
Your task is to generate a total topic that represents the entire meeting, written as a short Korean phrase.

Instructions:
- Capture the overarching theme that ties the segments together, not just one segment's detail.
- Output a single phrase only.
- Do not include any preamble, numbering, or explanation.
- Write in Korean.

Segment Summaries:
{numbered_summaries}

Segment Topics:
{topics_list}

Total topic:""".strip()

        messages = [
            {"role": "system", "content": "당신은 회의 전체 주제를 한국어 명사구로 표현하는 전문가입니다."},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages)

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
