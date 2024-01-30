import os
from transformers import pipeline


sentiment_pipeline = pipeline(model="fergusq/finbert-finnsentiment")

messages = [
    "This is a positive message in English.",
    "This is a negative message in English."
]

for message in messages:
    result = sentiment_pipeline([message])
    print(message, result[0])

