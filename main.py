from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
#nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text_data = request.form.getlist('text')
    results = []

    for text in text_data:
        sentiment_scores = sia.polarity_scores(text)
        sentiment = sentiment_scores['compound']

        if sentiment >= 0.05:
            category = "Positive"
        elif sentiment <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"

        result = {
            'text': text,
            'sentiment': sentiment,
            'category': category
        }
        results.append(result)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
