# AI-Chatbot
Here is a complete, production-ready project for an Advanced NLP AI Chatbot using Hugging Face Transformers and Gradio for an interactive Web UI. It includes a structured project layout, source code, a README.md, and step-by-step instructions to upload it to GitHub.
nlp-ai-chatbot/
│── app.py             # Gradio Web Interface
│── chatbot.py         # Advanced NLP Engine (Transformers model)
│── requirements.txt   # Required Python libraries
│── .gitignore         # Prevents unnecessary files from uploading to GitHub
└── README.md          # Project documentation for GitHub
torch
transformers
accelerate
gradio
sentencepiece
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class NLPChatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """
        Initializes the NLP engine with a Hugging Face Transformer model.
        """
        print(f"Loading NLP model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device.upper()}!")

    def generate_response(self, user_message, history=None):
        """
        Generates a contextual response given the user prompt and conversation history.
        """
        if history is None:
            history = []

        # Format input prompt from history
        context = ""
        for user_text, bot_text in history:
            context += f"User: {user_text}\nBot: {bot_text}\n"
        context += f"User: {user_message}\nBot:"

        # Tokenize input
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

        # Decode generated text
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract bot's reply from full context
        response = full_response.split("Bot:")[-1].strip()
        return response

if __name__ == "__main__":
    bot = NLPChatbot()
    print("Test output:", bot.generate_response("Hello, how are you?"))
    import gradio as gr
from chatbot import NLPChatbot

# Initialize the Chatbot
bot = NLPChatbot()

def chat_function(message, history):
    """
    Callback function for Gradio ChatInterface.
    """
    return bot.generate_response(message, history)

# Build Gradio Chat Interface
demo = gr.ChatInterface(
    fn=chat_function,
    title="🤖 Advanced NLP AI Chatbot",
    description="An open-source conversational AI chatbot built with Hugging Face Transformers and PyTorch.",
    theme="soft",
    examples=[
        "Hello! Who are you?",
        "Can you explain how Natural Language Processing works?",
        "What are your favorite topics to talk about?"
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
    __pycache__/
*.pyc
*.pyo
*.pyd
.env
venv/
.vscode/
.idea/
# 🤖 Advanced NLP AI Chatbot

An open-source conversational AI chatbot built using **Hugging Face Transformers**, **PyTorch**, and **Gradio**.

---

## ✨ Features
- 🧠 **Transformer-powered**: Uses deep learning NLP architectures for context-aware dialog generation.
- 💬 **Interactive UI**: Simple, fast web interface powered by Gradio.
- ⚡ **CPU/GPU Auto-Detection**: Runs seamlessly on CPU or GPU hardware via PyTorch.
- 🛠️ **Customizable**: Easy to swap underlying models (e.g., Llama, Mistral, DialoGPT, Flan-T5).

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/nlp-ai-chatbot.git](https://github.com/YOUR_USERNAME/nlp-ai-chatbot.git)
cd nlp-ai-chatbot
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app.py
