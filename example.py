from flask import Flask, request, jsonify
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import joblib

app = Flask(__name__)

# Function to process comments and categorize them (replace with your logic)
def categorize_comments(video_url):
    # Process comments and categorize them
    # Replace this with your actual code for processing comments

    # Sample categorized comments

    api_service_name = 'youtube'
    api_version = 'v3'
    developer_key = 'AIzaSyDssIxO4pqiiTrkeIgm1_DjazziBlODkMY'

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=developer_key)


    comments = []

    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_url,
            maxResults=100,  # You can adjust this value as needed (up to 100 per request).
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['authorDisplayName'], comment['textDisplay']])
        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break
    df = pd.DataFrame(comments, columns=['author', 'text'])

    sen_df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    sen_text = sen_df['text'].tolist()

    svm_classifier = joblib.load('svm_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')

    sent_features = tfidf.transform(sen_text)

    predicted_labels = svm_classifier.predict(sent_features)
    sen_df['predicted_sentiment'] = predicted_labels

    positive_list = sen_df[sen_df['predicted_sentiment'] == 2]['text'].tolist()
    negative_list = sen_df[sen_df['predicted_sentiment'] == 0]['text'].tolist()
    neutral_list = sen_df[sen_df['predicted_sentiment'] == 1]['text'].tolist()

    print("Positive List:", positive_list)
    print("Negative List:", negative_list)
    print("Neutral List:", neutral_list)

    return {
        "positive_comments": positive_list,
        "negative_comments": negative_list,
        "neutral_comments": neutral_list,
    }

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        # Get the video URL from the POST request
        video_url = request.form.get('video_url')

        # Call the function to categorize comments
        categorized_comments = categorize_comments(video_url)

        # Return categorized comments in JSON format
        return jsonify(categorized_comments)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
