from transformers import pipeline
from lingua import Language, LanguageDetectorBuilder

models = {
    Language.ENGLISH: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.JAPANESE: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.ARABIC: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.GERMAN: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.SPANISH: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.FRENCH: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.CHINESE: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.INDONESIAN: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.HINDI: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.ITALIAN: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.MALAY: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.PORTUGUESE: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    Language.SWEDISH: "KBLab/robust-swedish-sentiment-multiclass",
    Language.FINNISH: "fergusq/finbert-finnsentiment",
}

language_detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
sentiment_pipeline = pipeline(model="fergusq/finbert-finnsentiment")

messages = [
    "I'm happy to write in English. This should not be hard to detect.",
    "I'm sad to write in English. This is long enough to be detected.",
    "Olen iloinen",
    "Harmillinen juttu",
    "Detta är inte bra",
]

for message in messages:
    language = language_detector.detect_language_of(message)
    if language in models:
        sentiment_model = pipeline(model=models[language])
        result = sentiment_model([message])
        print(message, language, result[0])

    else:
        print(message, language)
        print(f"No model set for {language}")

