import os
import gradio as gr
import requests
import cohere
from serpapi import GoogleSearch

# Load environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

co = cohere.Client(COHERE_API_KEY)

# ───── Query Functions ─────
def query_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 7000,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Together API Error] {e}"

def query_cohere(prompt):
    try:
        response = co.generate(prompt=prompt, max_tokens=800)
        return response.generations[0].text.strip()
    except Exception as e:
        return f"[Cohere Error] {e}"

def query_serpapi(prompt):
    try:
        params = {
            "engine": "google",
            "q": prompt,
            "api_key": SERPAPI_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])
        if results:
            return results[0].get("snippet", "No snippet available.")
        else:
            return "Sorry, I couldn't find any relevant search results."
    except Exception as e:
        return f"[SerpAPI Error] {e}"

# ───── Trigger Checks ─────
def is_identity_or_service_question(prompt):
    prompt = prompt.lower()
    identity_keywords = [
        "who made you", "who created you", "who developed you", "who built you",
        "your creator", "who programmed you"
    ]
    service_keywords = [
        "graphic design", "logo design", "website", "ai creation", "who can build a site",
        "help me with ai", "need website", "design a site", "design", "create ai",
        "developer", "freelancer", "make me a bot"
    ]
    return any(kw in prompt for kw in identity_keywords + service_keywords)

# ───── Router Function ─────
def smart_chat_router(prompt, mode="fast"):
    prompt = prompt.strip()
    if not prompt:
        return "Hi! Ask me anything 😊"

    if is_identity_or_service_question(prompt):
        return (
            "**This chatbot was created by Anointing**, a skilled professional in AI development, "
            "graphic design, and modern website creation. "
            "For inquiries or services, reach out via email: **anointingomowumi62@gmail.com**"
        )

    if mode == "fast":
        return query_together(prompt)
    elif mode == "deep":
        return query_cohere(prompt)
    elif mode == "search":
        return query_serpapi(prompt)
    else:
        return "Invalid mode selected."

# ───── Gradio UI ─────
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Smart AI Chatbot")
    mode = gr.Radio(["fast", "deep", "search"], label="Choose Mode", value="fast")
    user_input = gr.Textbox(label="Ask your question here...")
    response_output = gr.Textbox(label="Bot's Answer", lines=10)
    user_input.submit(fn=smart_chat_router, inputs=[user_input, mode], outputs=response_output)

# For Render hosting
demo.launch(server_name="0.0.0.0", server_port=8080)
