import os
import gradio as gr
import requests
import cohere

# Load API Keys
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
DEEPAI_API_KEY = os.environ.get("DEEPAI_API_KEY")

co = cohere.Client(COHERE_API_KEY)

# --- API Query Functions ---

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

# --- DeepAI Functions ---

def deepai_request(url, files=None, data=None):
    headers = {"api-key": DEEPAI_API_KEY}
    try:
        response = requests.post(url, headers=headers, files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def deepai_image_gen(prompt):
    return deepai_request("https://api.deepai.org/api/text2img", data={"text": prompt})

def deepai_style_transfer(content_img, style_img):
    return deepai_request(
        "https://api.deepai.org/api/neural-style",
        files={"content": content_img, "style": style_img}
    )

def deepai_colorization(img):
    return deepai_request("https://api.deepai.org/api/colorizer", files={"image": img})

def deepai_recognition(img):
    return deepai_request("https://api.deepai.org/api/image-similarity", files={"image1": img})

def deepai_text_summarize(text):
    return deepai_request("https://api.deepai.org/api/summarization", data={"text": text})

def deepai_text_gen(text):
    return deepai_request("https://api.deepai.org/api/text-generator", data={"text": text})

# --- Trigger Check ---

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

# --- Gradio App ---

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Smart AI Chatbot + DeepAI Tools")

    with gr.Tab("Chat"):
        mode = gr.Radio(["fast", "deep", "search", "summarize", "generate"], label="Choose Mode", value="fast")
        user_input = gr.Textbox(label="Ask your question here...")
        response_output = gr.Textbox(label="Bot's Answer", lines=10)
        user_input.submit(fn=lambda prompt, m: (
            "**This chatbot was created by Anointing**, expert in AI/web/design.\n📧 Contact: **anointingomowumi62@gmail.com**"
            if is_identity_or_service_question(prompt) else
            smart_chat_router(prompt, m)
        ), inputs=[user_input, mode], outputs=response_output)

    with gr.Tab("Image Generation"):
        text_prompt = gr.Textbox(label="Enter image description")
        img_out = gr.Image(label="Generated Image")
        gen_btn = gr.Button("Generate")
        gen_btn.click(
            fn=lambda p: deepai_image_gen(p).get("output_url", "Error generating image"),
            inputs=text_prompt,
            outputs=img_out
        )

    with gr.Tab("Style Transfer"):
        content = gr.Image(label="Content Image", type="file")
        style = gr.Image(label="Style Image", type="file")
        styled_out = gr.Image(label="Stylized Output")
        style_btn = gr.Button("Apply Style")
        style_btn.click(
            fn=lambda c, s: deepai_style_transfer(c, s).get("output_url", "Error"),
            inputs=[content, style],
            outputs=styled_out
        )

    with gr.Tab("Colorize"):
        gray_img = gr.Image(label="Upload B/W Image", type="file")
        colorized = gr.Image(label="Colorized Image")
        color_btn = gr.Button("Colorize")
        color_btn.click(
            fn=lambda i: deepai_colorization(i).get("output_url", "Error"),
            inputs=gray_img,
            outputs=colorized
        )

    with gr.Tab("Image Recognition"):
        input_img = gr.Image(label="Upload Image", type="file")
        result = gr.Textbox(label="Result")
        rec_btn = gr.Button("Analyze")
        rec_btn.click(
            fn=lambda i: deepai_recognition(i).get("output", "Error"),
            inputs=input_img,
            outputs=result
        )

# --- Launch for Render ---
demo.launch(server_name="0.0.0.0", server_port=8080)
