// static/script.js - Place this in a 'static' folder

const HEALTH_CONDITIONS = [
    "General Health",
    "Cardiovascular",
    "Respiratory",
    "Digestive",
    "Neurological",
    "Musculoskeletal",
    "Endocrine",
    "Mental Health"
];

let currentCondition = "General Health";
let currentAiMessageId = null;

// Initialize tabs
function initTabs() {
    const tabHeaders = document.querySelector('.tab-headers');
    const tabContent = document.querySelector('.tab-content');

    // Clear existing
    tabHeaders.innerHTML = '';
    tabContent.innerHTML = '';

    HEALTH_CONDITIONS.forEach((condition, index) => {
        // Tab header
        const header = document.createElement('div');
        header.className = 'tab-header';
        header.textContent = condition;
        header.onclick = () => switchTab(index);
        tabHeaders.appendChild(header);

        // Tab pane
        const pane = document.createElement('div');
        pane.id = `tab-${condition.toLowerCase().replace(/\s+/g, '-')}`;
        pane.className = 'tab-pane';
        if (index === 0) {
            pane.classList.add('active');
        }

        // Each tab has its own chat container
        const chatContainer = document.createElement('div');
        chatContainer.className = 'chat-container';
        chatContainer.innerHTML = `
            <div class="chat-messages" id="chat-messages-${index}"></div>
            <div class="input-area">
                <textarea id="message-input-${index}" placeholder="Type your message here..." rows="3"></textarea>
                <button id="send-btn-${index}">Send</button>
                <button id="clear-btn-${index}">Clear Chat</button>
            </div>
        `;
        pane.appendChild(chatContainer);
        tabContent.appendChild(pane);
    });

    // Set up event listeners for the first tab initially
    setupEventListeners(0);
}

function switchTab(index) {
    // Update active classes
    document.querySelectorAll('.tab-header').forEach((h, i) => {
        h.classList.toggle('active', i === index);
    });
    document.querySelectorAll('.tab-pane').forEach((p, i) => {
        p.classList.toggle('active', i === index);
    });

    // Update current condition
    currentCondition = HEALTH_CONDITIONS[index];

    // Setup event listeners for new active tab
    setupEventListeners(index);
}

function setupEventListeners(tabIndex) {
    const inputId = `message-input-${tabIndex}`;
    const sendBtnId = `send-btn-${tabIndex}`;
    const clearBtnId = `clear-btn-${tabIndex}`;
    const messagesId = `chat-messages-${tabIndex}`;

    const input = document.getElementById(inputId);
    const sendBtn = document.getElementById(sendBtnId);
    const clearBtn = document.getElementById(clearBtnId);
    const messages = document.getElementById(messagesId);

    // Send button
    sendBtn.onclick = () => sendMessage(tabIndex, input, messages);

    // Enter to send (allow shift+enter for new line)
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(tabIndex, input, messages);
        }
    });

    // Clear chat
    clearBtn.onclick = () => {
        messages.innerHTML = '';
    };
}

async function sendMessage(tabIndex, input, messages) {
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, 'user', messages);
    input.value = '';

    // Show thinking indicator
    showThinking();

    // Prepare AI message container
    currentAiMessageId = `ai-${Date.now()}`;
    addMessage('', 'ai', messages, currentAiMessageId);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, condition: currentCondition })
        });

        if (!response.ok) {
            throw new Error('Network error');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let aiMessage = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
                if (!line) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.delta) {
                        aiMessage += data.delta;
                        updateMessage(currentAiMessageId, aiMessage, messages);
                    } else if (data.done) {
                        // Stream complete
                        break;
                    } else if (data.error) {
                        throw new Error(data.error);
                    }
                } catch (e) {
                    console.error('Parse error:', e);
                }
            }
        }

        hideThinking();
    } catch (error) {
        console.error('Error:', error);
        updateMessage(currentAiMessageId, `Error: ${error.message}`, messages);
        hideThinking();
    }
}

function addMessage(text, type, container, id = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    if (id) messageDiv.id = id;
    messageDiv.textContent = text;
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

function updateMessage(id, text, container) {
    const message = document.getElementById(id);
    if (message) {
        message.textContent = text;
        container.scrollTop = container.scrollHeight;
    }
}

function showThinking() {
    document.getElementById('thinking-indicator').classList.remove('hidden');
}

function hideThinking() {
    document.getElementById('thinking-indicator').classList.add('hidden');
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initTabs);