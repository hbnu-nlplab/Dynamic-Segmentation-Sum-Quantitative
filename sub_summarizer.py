import os
import openai
from dotenv import load_dotenv


class SubTopicSummarizer:
    """이미 주어진 sub topic에 매칭된 세그먼트들을 하나로 묶어 topic 단위 요약을 생성한다."""

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

    def summarize_topic(self, topic, sentences):
        if not sentences:
            return ""

        text = " ".join(sentences)
        prompt = f"""You are an expert in writing summaries of meeting transcripts.

Below, you are given a meeting topic and a set of sentences collected from meeting utterances related to that topic.
Your task is to write a meeting summary based on the sentences, focusing on the given topic.

Topic:
{topic}

Instructions:
    1. Write in a concise and clear style, similar to news articles.
    2. The summary should be around 4 sentences.
    3. Only output the summary.
    4. Write in Korean.

Sentences:
{text}

Summary:""".strip()

        messages = [{"role": "user", "content": prompt}]
        return self._chat(messages)
