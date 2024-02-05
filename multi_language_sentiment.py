"""
Sentiment analysis pipeline for texts in multiple languages.
"""

import gc
from collections import defaultdict

from transformers import pipeline
from lingua import Language, LanguageDetectorBuilder


__version__ = "0.0.1"


default_models = {
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
    "Does this get detected?",
    "Olen iloinen",
    "Harmillinen juttu",
    "Detta är inte bra",
    "Jag skall gå till supermarket",
]


# Processing a batch:
# Detect languages into a list and map to models
# For each model, make a pipeline, make a list and process
# inject int a list in the original order

def process_messages_in_batches(messages_with_languages, models = None):
    """
    Process messages in batches, creating only one pipeline at a time, and maintain the original order.
    
    Params:
    messages_with_languages: list of tuples, each containing a message and its detected language
    models: dict, model paths indexed by Language
    
    Returns:
    OrderedDict: containing the index as keys and tuple of (message, sentiment result) as values
    """

    if models is None:
        models = default_models
    else:
        models = default_models.copy().update(models)

    # Group messages by model, preserving original order
    messages_by_model = defaultdict(list)
    for index, (message, language) in enumerate(messages_with_languages):
        model_name = models.get(language)
        if model_name:
            messages_by_model[model_name].append((index, message))

    # Process messages and maintain original order
    results = OrderedDict()
    for model_name, batch in messages_by_model.items():
        sentiment_pipeline = pipeline(model=model_name)
        batch_results = sentiment_pipeline([message for _, message in batch])
        del sentiment_pipeline

        for (index, _), sentiment_result in zip(batch, batch_results):
            results[index] = (batch[0][1], sentiment_result['label'])
    
    return results


messages_with_languages = [
    (message, language_detector.detect_language_of(message)) for message in messages
]
result_dict = process_messages_in_batches(messages_with_languages)
result = [result_dict[i] for i in range(len(result_dict))]

print(result)

