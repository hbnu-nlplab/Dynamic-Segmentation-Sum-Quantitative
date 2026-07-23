import os
import openai
from dotenv import load_dotenv


class SegmentSummarizer:
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

    def summarize_segment(self, sentences):
        if not sentences:
            return ""

        prompt = f"""You are an expert in meeting summarization.
Below is a segment consisting of semantically related utterances from a meeting.
Your task is to summarize the CONTENT of the segment, not the meeting process itself.

Instructions:
1. Write in a concise and clear style, similar to news articles.
2. The summary should be around 4 sentences.
3. Only output the summary.
4. Write in Korean.

Transcript Segment:
{" ".join(sentences)}

Summary:""".strip()

        messages = [
            {"role": "system", "content": "당신은 회의 구간(segment)을 요약하는 전문가입니다."},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages)
