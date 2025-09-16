import os
import requests
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Updated model - using a reliable free model
MODEL_NAME = "meta-llama/llama-3.2-3b-instruct:free"

# Alternative models you can try:
# MODEL_NAME = "microsoft/wizardlm-2-8x22b:nitro"  
# MODEL_NAME = "anthropic/claude-3.5-sonnet"
# MODEL_NAME = "openai/gpt-4o-mini"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CureBot - Your Medical Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            margin-bottom: 5px;
            font-size: 2.5em;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .chat-container {
            flex: 1;
            padding: 20px;
            display: flex;
            flex-direction: column;
            max-height: 60vh;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 20px;
            padding-right: 10px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 15px;
            max-width: 80%;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: #3498db;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: #ecf0f1;
            color: #2c3e50;
            border-left: 4px solid #27ae60;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 0 0 20px 20px;
        }
        
        #messageInput {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #messageInput:focus {
            border-color: #3498db;
        }
        
        #sendButton {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }
        
        #sendButton:hover {
            transform: translateY(-2px);
        }
        
        #sendButton:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 10px;
            color: #7f8c8d;
        }
        
        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 15px;
            margin: 15px;
            font-size: 0.9em;
            color: #856404;
        }
        
        .scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        
        .scrollbar::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .scrollbar::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        
        .scrollbar::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 CureBot</h1>
            <p>Your AI Medical Assistant</p>
        </div>
        
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This AI assistant provides general health information for educational purposes only. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment.
        </div>
        
        <div class="chat-container">
            <div id="messages" class="messages scrollbar">
                <div class="message bot-message">
                    👋 Hello! I'm CureBot, your medical assistant. I can help with general health questions, symptoms, and wellness tips. How can I assist you today?
                </div>
            </div>
            
            <div class="loading" id="loading">
                🤔 CureBot is thinking...
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Ask me about symptoms, health concerns, or wellness tips..." maxlength="500">
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('sendButton').disabled = true;
            
            // Send to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('sendButton').disabled = false;
                
                if (data.error) {
                    addMessage('Sorry, I encountered an error: ' + data.error, 'bot');
                } else {
                    addMessage(data.response, 'bot');
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('sendButton').disabled = false;
                addMessage('Sorry, there was a connection error. Please try again.', 'bot');
                console.error('Error:', error);
            });
        }
        
        function addMessage(text, type) {
            const messagesContainer = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = text;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Allow Enter key to send message
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

def create_medical_prompt(user_message):
    return f"""You are CureBot, a helpful medical assistant AI. Provide accurate, helpful information about health topics while being empathetic and professional.

IMPORTANT GUIDELINES:
- Always include a disclaimer that your advice doesn't replace professional medical care
- For serious symptoms, recommend seeing a healthcare provider
- Be compassionate and understanding
- Provide practical, actionable advice when appropriate
- If unsure about something medical, say so clearly

User Question: {user_message}

Please provide a helpful, informative response:"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        if not OPENROUTER_API_KEY:
            return jsonify({'error': 'OpenRouter API key not configured'}), 500
        
        # Create the medical prompt
        full_prompt = create_medical_prompt(user_message)
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://curebot.onrender.com",  # Replace with your actual domain
            "X-Title": "CureBot Medical Assistant"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Always be professional, empathetic, and include appropriate medical disclaimers."
                },
                {
                    "role": "user", 
                    "content": full_prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        # Make request to OpenRouter
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_detail = response.text
            print(f"OpenRouter API Error: {response.status_code} - {error_detail}")
            
            # Provide specific error messages
            if response.status_code == 404:
                return jsonify({'error': 'Model not available. Please check the model name or try a different model.'}), 500
            elif response.status_code == 401:
                return jsonify({'error': 'Invalid API key. Please check your OpenRouter API key.'}), 500
            elif response.status_code == 429:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 500
            else:
                return jsonify({'error': f'API Error: {response.status_code}'}), 500
        
        response_data = response.json()
        
        if 'choices' not in response_data or not response_data['choices']:
            return jsonify({'error': 'No response generated'}), 500
        
        bot_response = response_data['choices'][0]['message']['content']
        
        return jsonify({'response': bot_response})
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout. Please try again.'}), 500
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return jsonify({'error': 'Network error. Please try again.'}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model': MODEL_NAME})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))