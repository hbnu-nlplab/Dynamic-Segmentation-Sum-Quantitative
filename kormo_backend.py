import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("kormo_backend")


class KormoBackend:
    """학습이 끝나 base 모델에 병합된 KORMo-10B multitask 모델로 GPT 호출부를 대체하는 로컬 백엔드."""

    def __init__(self, model_dir, max_new_tokens=512, device_map="auto"):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        print(f"Kormo 로딩 중... (model_dir={self.model_dir})")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # bitsandbytes 4bit 로딩은 device_map이 단일 device 문자열이면 정상적으로
        # 양자화가 적용되지 않는 경우가 있어, 명시적으로 {"": device} 형태로 정규화한다
        device_map = self.device_map
        if isinstance(device_map, str) and device_map != "auto":
            device_map = {"": device_map}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quant_config,
        )
        self.model.eval()
        print("Kormo 로딩 완료")

    def _generate(self, user_prompt):
        self.load()

        messages = [{"role": "user", "content": user_prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
        except torch.cuda.OutOfMemoryError:
            logger.error("Kormo 생성 중 CUDA OOM 발생 → GPU 캐시를 비우고 예외를 전파합니다.")
            torch.cuda.empty_cache()
            raise

        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # 아래 4개 메서드는 원래 train.py의 프롬프트 템플릿과 동일하게 유지해야 LoRA 학습 분포와 맞지만,
    # generate_totalsum(sentence_range)과 generate_speakersum(문장 수)은 GPT 출력과 형태를 맞추기 위해
    # 의도적으로 문장 수 지시를 바꿨다 (학습 분포와 달라져 품질이 낮아질 수 있음을 감수).

    def generate_segsum(self, span_text_block):
        prompt = f"""
You are an expert in meeting summarization.
Below is a segment consisting of semantically related utterances from a meeting.

Your task is to summarize the CONTENT of the segment, not the meeting process itself.

Instructions:
1. Write in a concise and clear style, similar to news articles.
2. The summary should be around 4 sentences.
3. Only output the summary.
4. Write in Korean.

Transcript Segment:
{span_text_block}

Summary:
""".strip()
        return self._generate(prompt)

    def generate_subtopic(self, span_text_block, summary_text):
        prompt = f"""
You are an expert at expressing the sub topic of a conversation segment as a Korean noun phrase.
Below is a segment consisting of semantically related utterances from a meeting, along with its summary.

Your task is to generate a sub topic that represents this segment, written as a Korean noun phrase of 2 to 5 words.

Instructions:
1. Capture only the key content specific to this segment.
2. Output a single noun phrase only.
3. Do not include any preamble, numbering, or explanation.
4. Write in Korean.

Examples:
- 눈과 입을 통한 감정 표현 방식의 차이
- 역사 왜곡과 고증 오류의 차이
- 저출산에 따른 사회 변화
- 해외의 저출산 정책

Transcript Segment:
{span_text_block}

Segment Summary:
{summary_text}

Sub topic:
""".strip()
        return self._generate(prompt)

    def generate_totalsum(self, summaries, sentence_range="4~5"):
        concat = "\n".join(f"[Segment Summary {i}] {s}" for i, s in enumerate(summaries, 1))
        prompt = f"""
You are an expert in meeting summarization.
Below are segment-level summaries from one meeting.

Your task is to generate a total summary of the whole meeting based on the segment summaries.

Instructions:
1. Write in a concise and clear style, similar to news articles.
2. The summary should be around {sentence_range} sentences.
3. Only output the summary.
4. Write in Korean.

Segment Summaries:
{concat}

Total Summary:
""".strip()
        return self._generate(prompt)

    def generate_speakersum(self, span_text_block, speaker, speaker_utterance_block):
        prompt = f"""
You are an expert in meeting summarization.
Below is a segment consisting of semantically related utterances from a meeting, followed by the utterances of a single speaker within that segment.

Your task is to summarize what speaker "{speaker}" said in this segment, using the full segment as context.

Instructions:
1. Focus on the speaker's main claims, opinions, or reactions.
2. If the speaker only gave simple reactions with no substantial content, output "주요 내용 없음".
3. Write in 3 sentences.
4. Only output the summary.
5. Write in Korean.

Transcript Segment:
{span_text_block}

Utterances by {speaker}:
{speaker_utterance_block}

Speaker Summary:
""".strip()
        return self._generate(prompt)
