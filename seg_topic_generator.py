import os
import openai
from dotenv import load_dotenv


class SegTopicGenerator:
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

    def generate_topic(self, sentences, summary=""):
        if not sentences:
            return ""

        summary_section = f"\n[Segment Summary]\n{summary}\n" if summary else ""

        prompt = f"""Below is a segment of utterances from a single conversation.

Your task is to generate a sub topic that represents this segment, written as a Korean noun phrase of 2 to 5 words.

Instructions:
- Capture only the key content specific to this segment.
- Output a single noun phrase only.
- Do not include any preamble, numbering, or explanation.

Examples:
- sub topic : 눈과 입을 통한 감정 표현 방식의 차이
- sub topic : 역사 왜곡과 고증 오류의 차이
- sub topic : 저출산에 따른 사회 변화
- sub topic : 해외의 저출산 정책
{summary_section}
[Conversation Segment]
{" ".join(sentences)}

Sub topic:""".strip()

        messages = [
            {"role": "system", "content": "You are an expert at expressing the sub topic of a conversation segment as a Korean noun phrase."},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._chat(messages)
        except Exception as e:
            print(f"❌ GPT 실패 - topic 생성 | {e}")
            return "[ERROR]"
