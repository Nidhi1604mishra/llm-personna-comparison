# 🤖 LLM Persona & Model Comparison App

This project is a **proof of concept (PoC)** designed to evaluate how different large language models (LLMs) respond to the same query under various **personas** such as Customer Support, Technical Expert, and Creative Writer.

It leverages **LangChain** for prompt chaining, **OpenRouter API** for multi-model access, and **Gradio** for an interactive UI.

---

## 🚀 Features

- 🔄 Compare responses across multiple LLMs: GPT-3.5, Claude 3, Gemini Pro, Mistral, Cohere, DeepSeek, etc.
- 🎭 Support for multiple personas:
  - Customer Support Agent
  - IT Technical Expert
  - Creative Writer
- 🧠 Built using LangChain's prompt chaining mechanism
- ⏱️ Tracks response time for each model
- 🖥️ Interactive web UI built with Gradio

---

## 🔧 Tech Stack

| Tech             | Purpose                             |
|------------------|-------------------------------------|
| Python           | Core programming language           |
| LangChain        | Prompt chaining and LLM interfacing |
| OpenRouter API   | Unified API access to various LLMs  |
| Gradio           | Web-based interface for interaction |
| dotenv           | For loading API keys securely       |
| Matplotlib / Seaborn | Optional: for result visualization (if needed) |

---


---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-persona-comparison.git
cd llm-persona-comparison

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt

Open_api_key=your_openrouter_api_key_here

python app.py

Running on local URL:  http://127.0.0.1:7860
