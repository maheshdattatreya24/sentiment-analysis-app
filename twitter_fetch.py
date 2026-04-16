import random

def get_tweets(query, count=5):
    sample_tweets = [
        f"I love {query}, it's amazing!",
        f"{query} is the worst thing ever",
        f"Not sure about {query}, it's okay",
        f"{query} is fantastic!",
        f"I hate using {query}",
        f"{query} is trending and people love it",
        f"Bad experience with {query}",
        f"{query} is really helpful"
    ]

    return random.sample(sample_tweets, count)