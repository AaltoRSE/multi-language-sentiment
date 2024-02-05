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

    results = {}

    # Group messages by model, preserving original order.
    # If language is no detected or a model for that language is not
    # provided, add None to results
    messages_by_model = defaultdict(list)
    for index, (message, language) in enumerate(messages_with_languages):
        model_name = models.get(language)
        if model_name:
            messages_by_model[model_name].append((index, message))
        else:
            results[index] = {"label": "none", "score": 0}
            
    # Process messages and maintain original order
    for model_name, batch in messages_by_model.items():
        sentiment_pipeline = pipeline(model=model_name)
        batch_results = sentiment_pipeline([message for _, message in batch])

        # Force garbage collections to remove the model from memory
        del sentiment_pipeline
        gc.collect()

        for (index, _), sentiment_result in zip(batch, batch_results):
            results[index] = sentiment_result
    
    results = [results[i] for i in range(len(results))]

    # Unify common spellings of the labels
    for i in range(len(results)):
        results[i]["label"] = results[i]["label"].lower()

    return results


def sentiment(messages, models=None):
    """
    Estimate the sentiment of a list of messages (strings of text). The
    sentences may be in different languages from each other.

    We maintain a list of default models for some languages. In addition,
    the user can provide a model for a given language in the models
    dictionary. The keys for this dictionary are lingua.Language objects
    and items HuggingFace model paths.
    
    Params:
    messages: list of message strings
    models: dict, huggingface model paths indexed by lingua.Language
    
    Returns:
    OrderedDict: containing the index as keys and tuple of (message, sentiment result) as values
    """
    messages_with_languages = [
        (message, language_detector.detect_language_of(message)) for message in messages
    ]

    results = process_messages_in_batches(messages_with_languages, models)
    return  results

