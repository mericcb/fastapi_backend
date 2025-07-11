from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("OPENAI_API_KEY not found in .env file!")

client = openai.OpenAI(api_key=api_key)

app = FastAPI()

class FlashcardRequest(BaseModel):
    topic: str
    card_type: str
    count: int = Field(default=5, ge=1, le=10)
    language: str = Field(default="en", pattern="^(en|tr)$")

# English System Prompts
SYSTEM_PROMPT_EN = (
    "You are an expert educational content creator specializing in flashcards for students and lifelong learners. "
    "Your task is to create high-quality flashcards that contain the most important, essential facts about a topic (no trivia), "
    "clear and concise questions and answers that aid memory retention, and content useful for exams or real-life understanding. "
    "Keep questions and answers short and focused (max 25 words). "
    "Language level should be high school or university. "
    "Return the flashcards as a valid JSON array with no extra text. "
    "Each flashcard must have 'question' and 'answer' keys."
)

QUIZ_SYSTEM_PROMPT_EN = (
    "You are an expert educational content creator specializing in multiple-choice quizzes for students and lifelong learners. "
    "Your task is to create high-quality multiple-choice questions that cover the most important, essential facts about a topic (no trivia). "
    "Each question must have exactly 4 options and 1 correct answer. "
    "The options must be plausible and relevant. "
    "Language level should be high school or university. "
    "Return the result as a valid JSON array with no extra text. "
    "Each question must have 'question', 'options' (list of 4 strings), and 'correct_answer' (string) keys."
)

# Turkish System Prompts
SYSTEM_PROMPT_TR = (
    "Sen öğrenciler ve yaşam boyu öğrenenler için flashcard oluşturma konusunda uzman bir eğitim içeriği yaratıcısısın. "
    "Görevin bir konu hakkında en önemli, temel gerçekleri içeren (trivia değil), "
    "hafızayı güçlendiren net ve özlü sorular ve cevaplar ve sınavlar veya gerçek yaşam anlayışı için yararlı içerik içeren "
    "yüksek kaliteli flashcard'lar oluşturmaktır. "
    "Soruları ve cevapları kısa ve odaklanmış tut (maksimum 25 kelime). "
    "Dil seviyesi lise veya üniversite düzeyinde olmalı. "
    "Flashcard'ları ekstra metin olmadan geçerli bir JSON dizisi olarak döndür. "
    "Her flashcard'da 'question' ve 'answer' anahtarları bulunmalı."
)

QUIZ_SYSTEM_PROMPT_TR = (
    "Sen öğrenciler ve yaşam boyu öğrenenler için çoktan seçmeli sınavlar konusunda uzman bir eğitim içeriği yaratıcısısın. "
    "Görevin bir konu hakkında en önemli, temel gerçekleri kapsayan (trivia değil) yüksek kaliteli çoktan seçmeli sorular oluşturmaktır. "
    "Her soru tam olarak 4 seçeneğe ve 1 doğru cevaba sahip olmalı. "
    "Seçenekler makul ve konuyla ilgili olmalı. "
    "Dil seviyesi lise veya üniversite düzeyinde olmalı. "
    "Sonucu ekstra metin olmadan geçerli bir JSON dizisi olarak döndür. "
    "Her soruda 'question', 'options' (4 string'den oluşan liste), ve 'correct_answer' (string) anahtarları bulunmalı."
)

def get_system_prompt(card_type: str, language: str) -> str:
    """Get the appropriate system prompt based on card type and language"""
    if card_type == "flashcard":
        return SYSTEM_PROMPT_TR if language == "tr" else SYSTEM_PROMPT_EN
    elif card_type == "quiz":
        return QUIZ_SYSTEM_PROMPT_TR if language == "tr" else QUIZ_SYSTEM_PROMPT_EN
    else:
        raise ValueError("Invalid card_type")

def get_user_prompt(card_type: str, topic: str, count: int, language: str) -> str:
    """Get the appropriate user prompt based on card type and language"""
    if language == "tr":
        if card_type == "flashcard":
            return (
                f"'{topic}' konusu hakkında {count} adet flashcard oluştur. "
                f"Sonucu geçerli bir JSON dizisi olarak döndür. "
                f"Her flashcard'da 'question' ve 'answer' anahtarları bulunmalı. "
                f"Örnek: [{{\"question\": \"... nedir?\", \"answer\": \"...\"}}, ...]"
            )
        else:  # quiz
            return (
                f"'{topic}' konusu hakkında {count} adet çoktan seçmeli soru oluştur. "
                f"Sadece belirtilen geçerli JSON dizisini döndür."
            )
    else:  # English
        if card_type == "flashcard":
            return (
                f"Create {count} flashcards about '{topic}'. "
                f"Return the result as a valid JSON array. "
                f"Each flashcard should have 'question' and 'answer' keys. "
                f"Example: [{{\"question\": \"What is ...?\", \"answer\": \"...\"}}, ...]"
            )
        else:  # quiz
            return (
                f"Create {count} multiple-choice questions about '{topic}'. "
                f"Return only the valid JSON array as specified."
            )

@app.post("/generate")
async def generate_flashcard(req: FlashcardRequest):
    try:
        # Validate card_type
        if req.card_type not in ["flashcard", "quiz"]:
            raise HTTPException(status_code=400, detail="Invalid card_type. Use 'flashcard' or 'quiz'.")

        # Get appropriate prompts based on language
        system_prompt = get_system_prompt(req.card_type, req.language)
        user_prompt = get_user_prompt(req.card_type, req.topic, req.count, req.language)

        # ChatGPT API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        result_text = response.choices[0].message.content.strip()
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI did not return valid JSON")

        return {"result": result_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))