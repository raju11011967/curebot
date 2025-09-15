from flask import Flask, request, jsonify, render_template_string
import os
import requests

app = Flask(__name__)

# HTML template for the chat interface
CHAT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CureBot - Your Health Assistant</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { 
            color: #333; 
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .chat-container {
            height: 400px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 20px;
            background: #f9f9f9;
        }
        .message {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 20px;
            max-width: 70%;
        }
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: #e9ecef;
            color: #333;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        #messageInput {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
        }
        #sendButton {
            padding: 12px 25px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
        }
        #sendButton:hover { background: #0056b3; }
        #sendButton:disabled { background: #ccc; cursor: not-allowed; }
        .loading { color: #666; font-style: italic; }
        .error { color: #dc3545; }
        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 14px;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 CureBot</h1>
        <p class="subtitle">Your AI Health Assistant</p>
        
        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> CureBot provides general health information only. 
            Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
        </div>
        
        <div id="chatContainer" class="chat-container">
            <div class="message bot-message">
                👋 Hello! I'm CureBot, your AI health assistant. I can help with general health questions, 
                symptoms information, and wellness tips. How can I assist you today?
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Ask me about your health concerns..." 
                   onkeypress="handleKeyPress(event)">
            <button id="sendButton" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            const sendButton = document.getElementById('sendButton');
            const chatContainer = document.getElementById('chatContainer');
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            sendButton.disabled = true;
            sendButton.textContent = 'Sending...';
            
            // Add loading message
            const loadingId = addMessage('CureBot is thinking...', 'bot', 'loading');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                // Remove loading message
                document.getElementById(loadingId).remove();
                
                if (response.ok) {
                    addMessage(data.response, 'bot');
                } else {
                    addMessage(`Error: ${data.error || 'Something went wrong'}`, 'bot', 'error');
                }
            } catch (error) {
                document.getElementById(loadingId).remove();
                addMessage('Error: Could not connect to CureBot. Please try again.', 'bot', 'error');
            }
            
            sendButton.disabled = false;
            sendButton.textContent = 'Send';
        }
        
        function addMessage(message, sender, className = '') {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            const messageId = 'msg-' + Date.now();
            messageDiv.id = messageId;
            messageDiv.className = `message ${sender}-message ${className}`;
            messageDiv.textContent = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageId;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(CHAT_TEMPLATE)

@app.route('/api')
def api_info():
    return jsonify({
        'name': 'CureBot API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'GET /': 'Web chat interface',
            'GET /api': 'API information',
            'POST /chat': 'Send message to CureBot',
            'GET /health': 'Health check'
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the API key from environment variables
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            return jsonify({'error': 'CureBot is temporarily unavailable. API key not configured.'}), 500
        
        # Get user message from request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Make request to OpenRouter API
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://curebot.onrender.com',
            'X-Title': 'CureBot'
        }
        
        payload = {
            'model': 'openai/gpt-3.5-turbo',
            'messages': [
                {
                    'role': 'system', 
                    'content': '''You are CureBot, a helpful and empathetic AI health assistant. Follow these guidelines:

1. Provide accurate, helpful health information
2. Always recommend consulting healthcare professionals for serious concerns, diagnosis, or treatment
3. Be empathetic and supportive
4. Use clear, easy-to-understand language
5. Include relevant disclaimers when appropriate
6. Focus on general wellness, symptom information, and health education
7. Never provide specific medical diagnoses or replace professional medical advice

Remember: You're here to inform and support, not to diagnose or treat.'''
                },
                {'role': 'user', 'content': user_message}
            ],
            'max_tokens': 500,
            'temperature': 0.7
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            bot_response = response.json()['choices'][0]['message']['content']
            return jsonify({'response': bot_response})
        else:
            error_details = response.text
            print(f"OpenRouter API Error: {response.status_code} - {error_details}")
            return jsonify({'error': 'Sorry, CureBot is experiencing technical difficulties. Please try again later.'}), 500
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out. Please try again.'}), 500
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return jsonify({'error': 'Network error. Please check your connection and try again.'}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'CureBot',
        'version': '1.0.0',
        'timestamp': str(__import__('datetime').datetime.now())
    })

if __name__ == '__main__':
    app.run(debug=True)