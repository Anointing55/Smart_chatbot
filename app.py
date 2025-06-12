import os
import gradio as gr
import requests
import cohere

─── Load API Keys ───

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
DEEPAI_API_KEY = os.environ.get("DEEPAI_API_KEY")

co = cohere.Client(COHERE_API_KEY)

─── DeepAI Functions ───

def deepai_request(url, files=None, data=None): headers = {"api-key": DEEPAI_API_KEY} try: response = requests.post(url, headers=headers, files=files, data=data) return response.json() except Exception as e: return {"error": str(e)}

def deepai_text2img(prompt): return deepai_request("https://api.deepai.org/api/text2img", data={"text": prompt})

def deepai_style_transfer(content_path, style_path): files = {"content": open(content_path, "rb"), "style": open(style_path, "rb")} return deepai_request("https://api.deepai.org/api/neural-style", files=files)

def deepai_colorize(image_path): files = {"image": open(image_path, "rb")} return deepai_request("https://api.deepai.org/api/colorizer", files=files)

def deepai_summarize(prompt): return deepai_request("https://api.deepai.org/api/summarization", data={"text": prompt})

def deepai_generate(prompt): return deepai_request("https://api.deepai.org/api/text-generator", data={"text": prompt})

def deepai_image_recognition(image_path): files = {"image": open(image_path, "rb")} return deepai_request("https://api.deepai.org/api/image-similarity", files=files)

─── Query Functions ───

def query_together(prompt): url = "https://api.together.xyz/v1/chat/completions" headers = { "Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json" } payload = { "model": "meta-llama/Llama-3-8b-chat-hf", "messages": [{"role": "user", "content": prompt}], "max_tokens": 7000, "temperature": 0.7 } try: response = requests.post(url, headers=headers, json=payload) result = response.json() return result["choices"][0]["message"]["content"] except Exception as e: return f"[Together API Error] {e}"

def query_cohere(prompt): try: response = co.generate(prompt=prompt, max_tokens=800) return response.generations[0].text.strip() except Exception as e: return f"[Cohere Error] {e}"

def query_serpapi(prompt): if not SERPAPI_KEY: return "[SerpAPI Error] API key not set."

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

─── Trigger Check ───

def is_identity_or_service_question(prompt): prompt = prompt.lower() identity_keywords = [ "who made you", "who created you", "who developed you", "who built you", "your creator", "who programmed you" ] service_keywords = [ "graphic design", "logo design", "website", "ai creation", "who can build a site", "help me with ai", "need website", "design a site", "design", "create ai", "developer", "freelancer", "make me a bot", "contact", "how can i reach you" ] return any(kw in prompt for kw in identity_keywords + service_keywords)

─── Smart Router ───

def smart_chat_router(prompt, mode="fast"): prompt = prompt.strip() if not prompt: return "Hi! Ask me anything 😊"

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
elif mode == "summary":
    return deepai_summarize(prompt).get("output", "[Summarization Error]")
elif mode == "gen":
    return deepai_generate(prompt).get("output", "[Text Generation Error]")
else:
    return "Invalid mode selected."

─── Gradio UI ───

with gr.Blocks() as demo: gr.Markdown("## 🤖 Smart AI Chatbot") mode = gr.Radio(["fast", "deep", "search", "summary", "gen"], label="Choose Mode", value="fast") user_input = gr.Textbox(label="Ask your question here...") response_output = gr.Textbox(label="Bot's Answer", lines=10) user_input.submit(fn=smart_chat_router, inputs=[user_input, mode], outputs=response_output)

gr.Markdown("### 🖼️ DeepAI Visual Tools")
with gr.Row():
    text2img_input = gr.Textbox(label="Text to Image Prompt")
    text2img_output = gr.Image(label="Generated Image")
    text2img_input.submit(
        lambda p: deepai_text2img(p).get("output_url", "[Text2Img Error]"),
        inputs=text2img_input,
        outputs=text2img_output
    )

with gr.Row():
    content_img = gr.Image(label="Content Image")
    style_img = gr.Image(label="Style Image")
    stylized_output = gr.Image(label="Stylized Image")
    gr.Button("Apply Style Transfer").click(
        lambda c, s: deepai_style_transfer(c, s).get("output_url", "[Style Transfer Error]"),
        inputs=[content_img, style_img],
        outputs=stylized_output
    )

with gr.Row():
    color_input = gr.Image(label="Black & White Image")
    color_output = gr.Image(label="Colorized Image")
    gr.Button("Colorize").click(
        lambda img: deepai_colorize(img).get("output_url", "[Colorization Error]"),
        inputs=color_input,
        outputs=color_output
    )

with gr.Row():
    recog_input = gr.Image(label="Image to Recognize")
    recog_output = gr.Textbox(label="Recognition Result")
    gr.Button("Recognize").click(
        lambda img: deepai_image_recognition(img).get("output", "[Recognition Error]"),
        inputs=recog_input,
        outputs=recog_output
    )

─── Launch (Render Settings) ───

demo.launch(server_name="0.0.0.0", server_port=8080)
