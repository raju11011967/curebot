from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return {'message': 'Hello from Flask API!'}

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

@app.route('/')
def hello():
    return {'message': 'CureBot API is running!', 'status': 'healthy'}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the API key from environment variables
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key not configured'}), 500
        
        # Get user message from request
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Make request to OpenRouter API
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'openai/gpt-3.5-turbo',  # or your preferred model
            'messages': [
                {'role': 'system', 'content': 'You are CureBot, a helpful medical assistant. Provide helpful health information but always recommend consulting healthcare professionals for serious concerns.'},
                {'role': 'user', 'content': user_message}
            ]
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            bot_response = response.json()['choices'][0]['message']['content']
            return jsonify({'response': bot_response})
        else:
            return jsonify({'error': 'Failed to get response from AI'}), 500
            
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'CureBot'})

if __name__ == '__main__':
    app.run(debug=True)