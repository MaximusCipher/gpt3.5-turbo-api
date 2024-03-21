from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Load the classifier
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Load the vectorizer
vectorizer = joblib.load('vectorizer.joblib')

openai_client = OpenAI()

@app.route('/send-message', methods=['POST'])
def send_message():
    user_input = request.json.get('message')

    # Preprocess the user input
    input_features = vectorizer.transform([user_input])

    # Make prediction using the loaded classifier
    predicted_class = classifier.predict(input_features)[0]

    # If the predicted class corresponds to a specific intent, handle it using the trained model
    # Otherwise, use the GPT-3.5 Turbo API for generating responses
    if predicted_class == "some_specific_intent":
        # Handle the intent using the trained model
        bot_response = "Response based on the trained model"
    else:
        # Use GPT-3.5 Turbo API for generating responses
        response = openai_client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal::94FoHzEH",
            messages=[
                {"role": "system", "content": "GabAi is an AI dedicated to assisting victims of workplace discrimination in the Philippines, providing legal guidance and support. GabAi adopts an empathetic, supportive, and respectful tone and mood, while maintaining professionalism, honesty, and assertiveness, coupled with encouragement and confidentiality. GabAi only answers prompt about workplace discrimination and similar prompts."}, {"role": "user", "content": user_input},
            ],
            temperature=1,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        bot_response = response.choices[0].message.content

    return jsonify({'message': bot_response})

if __name__ == "__main__":
    app.run(debug=True)