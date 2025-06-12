import os
import gradio as gr
import requests
import cohere

# ─── Load API Keys ───
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
DEEPAI_API_KEY = os.environ.get("DEEPAI_API_KEY")

co = cohere.Client(COHERE_API_KEY)

# ─── DeepAI API Helpers ───
def deepai_textgen(prompt):
    try:
        res = requests.post(
            "https://api.deepai.org/api/text-generator",
            data={"text": prompt},
            headers={"api-key": DEEPAI_API_KEY},
        )
        return res.json().get("output", "No response.")
    except Exception as e:
        return f"[DeepAI TextGen Error] {e}"

def deepai_summarize(text):
    try:
        res = requests.post(
            "https://api.deepai.org/api/summarization",
            data={"text": text},
            headers={"api-key": DEEPAI_API_KEY},
        )
        return res.json().get("output", "No summary.")
    except Exception as e:
        return f"[DeepAI Summarize Error] {e}"

def deepai_text2img(prompt):
    try:
        res = requests.post(
            "https://api.deepai.org/api/text2img",
            data={"text": prompt},
            headers={"api-key": DEEPAI_API_KEY},
        )
        return res.json().get("output_url", "Error generating image.")
    except Exception as e:
        return f"[DeepAI Text2Img Error] {e}"

def deepai_style_transfer(content_file, style_file):
    try:
        res = requests.post(
            "https://api.deepai.org/api/neural-style",
            files={"content": content_file, "style": style_file},
            headers={"api-key": DEEPAI_API_KEY},
        )
        return res.json()
    except Exception as e:
        return {"output_url": f"[Style Transfer Error] {e}"}

def deepai_colorization(image_file):
    try:
        res = requests.post(
            "https://api.deepai.org/api/colorizer",
            files={"image": image_file},
            headers={"api-key": DEEPAI_API_KEY},
        )
        return res.json()
    except Exception as e:
        return {"output_url": f"[Colorization Error] {e}"}

def deepai_recognition(image_file):
    try:
        res = requests.post(
            "https://api.deepai.org/api/image-similarity",
            files={"image": image_file},
            headers={"api-key": DEEPAI_API_KEY},
        )
        return res.json()
    except Exception as e:
        return {"output": f"[Recognition Error] {e}"}

# ─── Together/Cohere/SerpAPI Logic ───
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
    gr.Markdown("## 🤖 Smart AI Chatbot with DeepAI Features")
    with gr.Tab("Chatbot"):
        mode = gr.Radio(["fast", "deep", "search"], label="Choose Mode", value="fast")
        user_input = gr.Textbox(label="Ask your question here...")
        response_output = gr.Textbox(label="Bot's Answer", lines=10)
        user_input.submit(fn=smart_chat_router, inputs=[user_input, mode], outputs=response_output)

    with gr.Tab("Text Generator"):
        tg_prompt = gr.Textbox(label="Enter prompt")
        tg_output = gr.Textbox(label="Generated text")
        gr.Button("Generate").click(fn=deepai_textgen, inputs=tg_prompt, outputs=tg_output)

    with gr.Tab("Summarizer"):
        sum_input = gr.Textbox(label="Paste text to summarize", lines=8)
        sum_output = gr.Textbox(label="Summary")
        gr.Button("Summarize").click(fn=deepai_summarize, inputs=sum_input, outputs=sum_output)

    with gr.Tab("Text to Image"):
        t2i_prompt = gr.Textbox(label="Describe the image")
        t2i_output = gr.Image(label="Generated Image")
        gr.Button("Generate Image").click(fn=deepai_text2img, inputs=t2i_prompt, outputs=t2i_output)

    with gr.Tab("Style Transfer"):
        content = gr.Image(label="Content Image", type="filepath")
        style = gr.Image(label="Style Image", type="filepath")
        styled_out = gr.Image(label="Stylized Output")
        style_btn = gr.Button("Apply Style")
        style_btn.click(
            fn=lambda c, s: deepai_style_transfer(open(c, "rb"), open(s, "rb")).get("output_url", "Error"),
            inputs=[content, style],
            outputs=styled_out
        )

    with gr.Tab("Colorize"):
        gray_img = gr.Image(label="Upload B/W Image", type="filepath")
        colorized = gr.Image(label="Colorized Image")
        color_btn = gr.Button("Colorize")
        color_btn.click(
            fn=lambda i: deepai_colorization(open(i, "rb")).get("output_url", "Error"),
            inputs=gray_img,
            outputs=colorized
        )

    with gr.Tab("Image Recognition"):
        input_img = gr.Image(label="Upload Image", type="filepath")
        result = gr.Textbox(label="Result")
        rec_btn = gr.Button("Analyze")
        rec_btn.click(
            fn=lambda i: deepai_recognition(open(i, "rb")).get("output", "Error"),
            inputs=input_img,
            outputs=result
        )

# ─── Launch ───
demo.launch(server_name="0.0.0.0", server_port=8080)
