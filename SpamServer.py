from flask import Flask, request, jsonify
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import joblib

app = Flask(__name__)


# Function to process comments and categorize them (replace with your logic)
def spam_comments(video_url):
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

    spam_df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    spam_text = spam_df['text'].tolist()

    spam_model = joblib.load('spam_model.pkl')
    spam_vectorizer = joblib.load('spam_vectorizer.pkl')

    spam_features = spam_vectorizer.transform(spam_text)

    predicted_labels = spam_model.predict(spam_features)
    spam_df['predicted_sentiment'] = predicted_labels

    spam_list = spam_df[spam_df['predicted_sentiment'] == 'Spam Comment']['text'].tolist()
    nonSpam_list = spam_df[spam_df['predicted_sentiment'] == 'Not Spam']['text'].tolist()

    print("Spam List:", spam_list)
    print("Non Spam List:", nonSpam_list)

    return {
        "Spam_comments": spam_list,
        "NonSpam_comments": nonSpam_list,
    }


@app.route('/analyze_spam_video', methods=['POST'])
def analyze_video():
    try:
        # Get the video URL from the POST request
        video_url = request.form.get('video_url')

        # Call the function to categorize comments
        categorized_comments = spam_comments(video_url)

        # Return categorized comments in JSON format
        return jsonify(categorized_comments)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
