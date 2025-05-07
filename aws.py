import boto3

client = boto3.client('comprehend', region_name='eu-central-1')

text = "The economic outlook is very promising and investor confidence is growing."

response = client.detect_sentiment(Text=text, LanguageCode='en')

print(response['Sentiment'])         # Eredmény: POSITIVE, NEGATIVE, NEUTRAL, MIXED
print(response['SentimentScore'])    # Részletes pontszámok
