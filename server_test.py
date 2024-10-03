import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Dict, Any
from wsgiref.simple_server import make_server

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Load and preprocess reviews data
reviews_df = pd.read_csv("data/reviews.csv")
reviews_df["Timestamp"] = pd.to_datetime(
    reviews_df["Timestamp"], format="%Y-%m-%d %H:%M:%S"
)
reviews = reviews_df.to_dict("records")


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(
        self, environ: Dict[str, Any], start_response: Callable[..., Any]
    ) -> bytes:
        """
        The environ parameter is a dictionary containing useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        def get_query_params(environ):
            query_string = environ.get("QUERY_STRING", "")
            return parse_qs(query_string)

        if environ["REQUEST_METHOD"] == "GET":
            params = get_query_params(environ)
            location = params.get("location", [None])[0]
            start_date = params.get("start_date", [None])[0]
            end_date = params.get("end_date", [None])[0]

            filtered_df = reviews_df.copy()

            if location:
                if location not in ALLOWED_LOCATIONS:
                    start_response(
                        "400 Bad Request", [("Content-Type", "application/json")]
                    )
                    return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

                filtered_df = filtered_df[filtered_df["Location"] == location]

            if start_date:
                try:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                    filtered_df = filtered_df[filtered_df["Timestamp"] >= start_date]
                except ValueError:
                    start_response(
                        "400 Bad Request", [("Content-Type", "application/json")]
                    )
                    return [
                        json.dumps({"error": "Invalid start date format"}).encode(
                            "utf-8"
                        )
                    ]

            if end_date:
                try:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                    filtered_df = filtered_df[filtered_df["Timestamp"] <= end_date]
                except ValueError:
                    start_response(
                        "400 Bad Request", [("Content-Type", "application/json")]
                    )
                    return [
                        json.dumps({"error": "Invalid end date format"}).encode("utf-8")
                    ]

            reviews = []
            for _, row in filtered_df.iterrows():
                sentiment = self.analyze_sentiment(row["ReviewBody"])
                review = {
                    "ReviewId": row["ReviewId"],
                    "ReviewBody": row["ReviewBody"],
                    "Location": row["Location"],
                    "Timestamp": row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment": sentiment,
                }
                reviews.append(review)

            reviews.sort(key=lambda x: x["sentiment"]["compound"], reverse=True)

            response_body = json.dumps(reviews, indent=2).encode("utf-8")
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ],
            )
            return [response_body]

        elif environ["REQUEST_METHOD"] == "POST":
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            post_data = environ["wsgi.input"].read(content_length).decode("utf-8")
            post_params = parse_qs(post_data)

            location = post_params.get("Location", [None])[0]
            review_body = post_params.get("ReviewBody", [None])[0]

            if not location or not review_body:
                start_response(
                    "400 Bad Request", [("Content-Type", "application/json")]
                )
                return [
                    json.dumps(
                        {"error": "Location and ReviewBody are required"}
                    ).encode("utf-8")
                ]

            if location not in ALLOWED_LOCATIONS:
                start_response(
                    "400 Bad Request", [("Content-Type", "application/json")]
                )
                return [json.dumps({"error": "Invalid location"}).encode("utf-8")]

            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp,
            }

            response_body = json.dumps(review, indent=2).encode("utf-8")
            start_response(
                "201 Created",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ],
            )
            return [response_body]

        else:
            start_response(
                "405 Method Not Allowed", [("Content-Type", "application/json")]
            )
            return [json.dumps({"error": "Method not allowed"}).encode("utf-8")]


ALLOWED_LOCATIONS = [
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona",
]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get("PORT", 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
