import os
import openai
from dotenv import load_dotenv


class SpeakerSummarizer:
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

    def summarize_speaker(self, seg_id, speaker, utterances, full_segment):
        prompt = f"""아래는 하나의 대화 구간(segment {seg_id}) 전체 내용과, 그 중 화자 "{speaker}"의 발화 문장들입니다.
전체 문맥을 참고하여 해당 화자가 이 구간에서 어떤 내용을 말했는지 1문장으로 요약해 주세요.

요약 조건:
- 화자의 주요 주장·의견·반응을 중심으로 작성하세요.
- 단순 호응(응, 맞아 등)만 있는 경우 "주요 내용 없음"으로 출력하세요.
- 머리말 없이 요약문만 출력하세요.

[전체 구간 문맥]
{" ".join(full_segment)}

[화자 "{speaker}"의 발화]
{" ".join(utterances)}"""

        messages = [
            {"role": "system", "content": "당신은 특정 화자의 발화 내용을 간결하게 요약하는 전문가입니다."},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._chat(messages)
        except Exception as e:
            print(f"❌ GPT 실패 - seg={seg_id} speaker={speaker} | {e}")
            return "[ERROR] GPT 호출 실패"

    def summarize_speaker_for_topic(self, topic, speaker, utterances, all_items):
        merged_text = " ".join(utterances)
        full_context = "\n".join(f"{spk}: {sent}" for spk, sent in all_items)

        prompt = f"""아래는 세부 주제"{topic}" 대화 구간 전체 내용과, 그 중 화자 "{speaker}"의 발화 문장들입니다.
전체 문맥을 참고하여 해당 화자가 이 구간에서 어떤 내용을 말했는지 1문장으로 요약해 주세요.

요약 조건:
- 화자의 주요 주장·의견·반응을 중심으로 작성하세요.
- 머리말 없이 요약문만 출력하세요.
- 단순 호응(응, 맞아 등)만 있는 경우 "주요 내용 없음"으로 출력하세요.

[전체 구간 문맥]
{full_context}

[화자 "{speaker}"의 발화]
{merged_text}

Summary:""".strip()

        messages = [
            {"role": "system", "content": "당신은 토픽 내 화자 발언을 요약하는 전문가입니다."},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._chat(messages)
        except Exception as e:
            print(f"❌ GPT 실패 - topic={topic} speaker={speaker} | {e}")
            return "[ERROR] GPT 호출 실패"
