import os
import gradio as gr
import requests
import cohere

# ─── Load API Keys ───
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

co = cohere.Client(COHERE_API_KEY)

# ─── Query Functions ───
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
    if not SERPAPI_KEY:
        return "[SerpAPI Error] API key not set."

    try:
        params = {
            "engine": "google",
            "q": prompt,
            "api_key": SERPAPI_KEY,
            "num": 5
        }
        response = requests.get("https://serpapi.com/search", params=params)
        result = response.json()

        if "error" in result:
            return f"[SerpAPI Error] {result['error']}"

        organic_results = result.get("organic_results", [])
        if not organic_results:
            return "🔍 No relevant results found."

        snippets = []
        for res in organic_results[:3]:
            title = res.get("title", "No title")
            snippet = res.get("snippet", "No snippet.")
            link = res.get("link", "")
            snippets.append(f"**{title}**\n{snippet}\n🔗 {link}\n")

        return "🔍 **Top Google Results**:\n\n" + "\n---\n".join(snippets)

    except Exception as e:
        return f"[SerpAPI Exception] {e}"

# ─── Trigger Check ───
def is_identity_or_service_question(prompt):
    prompt = prompt.lower()
    identity_keywords = [
        "who made you", "who created you", "who developed you", "who built you",
        "your creator", "who programmed you"
    ]
    service_keywords = [
        "graphic design", "logo design", "website", "ai creation", "who can build a site",
        "help me with ai", "need website", "design a site", "design", "create ai",
        "developer", "freelancer", "make me a bot", "contact", "how can i reach you"
    ]
    return any(kw in prompt for kw in identity_keywords + service_keywords)

# ─── Smart Router ───
def smart_chat_router(prompt, mode="fast"):
    prompt = prompt.strip()
    if not prompt:
        return "Hi! Ask me anything 😊"

    if is_identity_or_service_question(prompt):
        return (
            "**This chatbot was created by Anointing**, an expert in AI development, "
            "graphic design, and website creation.\n\n"
            "📧 Contact: **anointingomowumi62@gmail.com**"
        )

    if mode == "fast":
        return query_together(prompt)
    elif mode == "deep":
        return query_cohere(prompt)
    elif mode == "search":
        return query_serpapi(prompt)
    else:
        return "Invalid mode selected."

# ─── Gradio UI ───
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Smart AI Chatbot")
    mode = gr.Radio(["fast", "deep", "search"], label="Choose Mode", value="fast")
    user_input = gr.Textbox(label="Ask your question here...")
    response_output = gr.Textbox(label="Bot's Answer", lines=10)
    user_input.submit(fn=smart_chat_router, inputs=[user_input, mode], outputs=response_output)

# ─── Launch (Render Settings) ───
demo.launch(server_name="0.0.0.0", server_port=8080)
