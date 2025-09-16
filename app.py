import os
from flask import Flask, request, jsonify, render_template
import requests
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)

@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and communicate with OpenRouter API"""
    try:
        # Check if API key is loaded from environment variables
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not found in environment variables")
            print("Available env vars:", list(os.environ.keys()))
            return jsonify({"error": "API key not configured. Please set OPENROUTER_API_KEY in environment variables."}), 500
        
        print(f"API Key loaded: {'Yes' if api_key else 'No'}")
        print(f"API Key starts with: {api_key[:10]}..." if api_key else "No key")
        
        # Get user message from request
        user_message = request.json.get('message', '') if request.json else ''
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"User message received: {user_message[:100]}..." if len(user_message) > 100 else user_message)
        
        # Prepare headers for OpenRouter API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://curebot.onrender.com",  # Optional: helps with rate limiting
            "X-Title": "CureBot"  # Optional: for OpenRouter analytics
        }
        
        # Prepare data for OpenRouter API
        data = {
            "model": "anthropic/claude-3.5-sonnet",  # Using Claude 3.5 Sonnet (more stable than 3.7)
            "messages": [
                {
                    "role": "system",
                    "content": "You are CureBot, a helpful medical assistant. Provide informative and supportive responses about health topics, but always remind users to consult with healthcare professionals for medical advice."
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        print(f"Attempting to use model: {data['model']}")
        
        # Make request to OpenRouter API
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"OpenRouter response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"OpenRouter API Error: {response.status_code}")
            print(f"Error response: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Unknown error')
            except:
                error_message = response.text
            
            return jsonify({
                "error": f"OpenRouter API Error ({response.status_code})",
                "details": error_message
            }), 500
        
        # Parse successful response
        response_data = response.json()
        print(f"Response data keys: {list(response_data.keys())}")
        
        # Extract the AI response
        if 'choices' in response_data and len(response_data['choices']) > 0:
            ai_message = response_data['choices'][0]['message']['content']
            print(f"AI response: {ai_message[:100]}...")
            return jsonify({"response": ai_message})
        else:
            print(f"Unexpected response format: {response_data}")
            return jsonify({"error": "Unexpected response format from OpenRouter"}), 500
    
    except requests.exceptions.Timeout:
        print("Request timed out")
        return jsonify({"error": "Request timed out. Please try again."}), 504
    
    except requests.exceptions.ConnectionError:
        print("Connection error to OpenRouter")
        return jsonify({"error": "Could not connect to AI service. Please try again."}), 503
    
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {str(e)}")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500
    
    except ValueError as e:
        print(f"JSON parsing error: {str(e)}")
        return jsonify({"error": "Invalid response from AI service"}), 500
    
    except Exception as e:
        print(f"Unexpected exception: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    return jsonify({
        "status": "healthy",
        "api_key_configured": bool(api_key)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)