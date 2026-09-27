import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------

app = Flask(
    __name__,
    static_folder=STATIC_DIR
)

# --------------------------------------------------
# NVIDIA API
# --------------------------------------------------

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

NVIDIA_IMAGE_URL = (
    "https://ai.api.nvidia.com/v1/genai/"
    "black-forest-labs/flux.1-schnell"
)

# --------------------------------------------------
# API KEYS
# --------------------------------------------------

API_KEYS = [
    os.getenv("NVIDIA_API_KEY_1"),
    os.getenv("NVIDIA_API_KEY_2"),
    os.getenv("NVIDIA_API_KEY_3"),
    os.getenv("NVIDIA_API_KEY_4"),
    os.getenv("NVIDIA_API_KEY_5"),
    os.getenv("NVIDIA_API_KEY_6"),
]

# Empty keys remove karo
API_KEYS = [
    key.strip()
    for key in API_KEYS
    if key and key.strip()
]

key_index = 0


def get_key():
    global key_index

    if not API_KEYS:
        raise RuntimeError(
            "No NVIDIA API key configured. "
            "Add NVIDIA_API_KEY_1 in Render Environment Variables."
        )

    key = API_KEYS[key_index % len(API_KEYS)]
    key_index += 1

    return key


def make_headers(key):
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# --------------------------------------------------
# TEXT MODELS
# --------------------------------------------------

TEXT_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-20b",
    "mistralai/mistral-nemotron",
    "meta/muse-glimmer-30b",
    "moonshotai/kimi-k3",
]

# Vision model
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"


# --------------------------------------------------
# NVIDIA CHAT FUNCTION
# --------------------------------------------------

def ask_nvidia(messages, model=None, max_tokens=1200):

    if not API_KEYS:
        return {
            "ok": False,
            "error": (
                "No NVIDIA API keys configured. "
                "Add NVIDIA_API_KEY_1 in Render."
            )
        }

    models = [model] if model else TEXT_MODELS

    last_error = "Unknown NVIDIA API error"

    for current_model in models:

        for _ in range(len(API_KEYS)):

            key = get_key()

            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": max_tokens,
                "stream": False,
            }

            try:

                response = requests.post(
                    NVIDIA_CHAT_URL,
                    headers=make_headers(key),
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:

                    data = response.json()

                    choices = data.get("choices", [])

                    if choices:

                        message = choices[0].get(
                            "message",
                            {}
                        )

                        answer = message.get(
                            "content",
                            ""
                        )

                        if answer:

                            return {
                                "ok": True,
                                "answer": answer,
                                "model": current_model,
                            }

                    last_error = (
                        "NVIDIA returned an empty response."
                    )

                else:

                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            except requests.exceptions.Timeout:

                last_error = (
                    "NVIDIA API request timed out."
                )

            except requests.exceptions.RequestException as e:

                last_error = str(e)

            except Exception as e:

                last_error = str(e)

    return {
        "ok": False,
        "error": last_error,
    }


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    index_file = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    if not os.path.exists(index_file):

        return jsonify({
            "ok": False,
            "error": (
                "index.html not found. "
                "Create static/index.html"
            )
        }), 500

    return send_from_directory(
        STATIC_DIR,
        "index.html"
    )


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

@app.route("/static/<path:filename>")
def static_files(filename):

    return send_from_directory(
        STATIC_DIR,
        filename
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "online",
        "keys_configured": len(API_KEYS),
        "text_models": TEXT_MODELS,
        "vision_model": VISION_MODEL,
    })


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(
            force=True,
            silent=True
        )

        if not data:

            return jsonify({
                "ok": False,
                "error": "Invalid JSON."
            }), 400

        messages = data.get(
            "messages",
            []
        )

        if not isinstance(messages, list):

            return jsonify({
                "ok": False,
                "error": "messages must be a list."
            }), 400

        if not messages:

            return jsonify({
                "ok": False,
                "error": "No messages provided."
            }), 400

        # Last 20 messages only
        messages = messages[-20:]

        result = ask_nvidia(
            messages,
            max_tokens=1200
        )

        if not result.get("ok"):

            return jsonify(result), 502

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# VISION
# --------------------------------------------------

@app.route("/api/vision", methods=["POST"])
def vision():

    try:

        data = request.get_json(
            force=True,
            silent=True
        )

        if not data:

            return jsonify({
                "ok": False,
                "error": "Invalid JSON."
            }), 400

        image = data.get("image")

        prompt = data.get(
            "prompt",
            "Describe this image accurately."
        )

        if not image:

            return jsonify({
                "ok": False,
                "error": "Image is required."
            }), 400

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    }
                ]
            }
        ]

        result = ask_nvidia(
            messages,
            model=VISION_MODEL,
            max_tokens=1000
        )

        if not result.get("ok"):

            return jsonify(result), 502

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# IMAGE GENERATION
# --------------------------------------------------

@app.route("/api/image", methods=["POST"])
def image_generation():

    try:

        data = request.get_json(
            force=True,
            silent=True
        )

        if not data:

            return jsonify({
                "ok": False,
                "error": "Invalid JSON."
            }), 400

        prompt = data.get(
            "prompt",
            ""
        ).strip()

        if not prompt:

            return jsonify({
                "ok": False,
                "error": "Prompt is required."
            }), 400

        payload = {
            "prompt": prompt,
            "steps": 4,
            "width": 1024,
            "height": 1024,
            "seed": 0,
        }

        last_error = (
            "Image generation failed."
        )

        if not API_KEYS:

            return jsonify({
                "ok": False,
                "error": (
                    "No NVIDIA API key configured."
                )
            }), 502

        for _ in range(len(API_KEYS)):

            key = get_key()

            try:

                response = requests.post(
                    NVIDIA_IMAGE_URL,
                    headers=make_headers(key),
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:

                    result = response.json()

                    # Format 1
                    image_b64 = result.get(
                        "image"
                    )

                    if image_b64:

                        return jsonify({
                            "ok": True,
                            "image":
                                "data:image/png;base64,"
                                + image_b64
                        })

                    # Format 2
                    artifacts = result.get(
                        "artifacts"
                    )

                    if artifacts:

                        b64 = artifacts[0].get(
                            "base64"
                        )

                        if b64:

                            return jsonify({
                                "ok": True,
                                "image":
                                    "data:image/png;base64,"
                                    + b64
                            })

                    last_error = (
                        "Image API returned no image."
                    )

                else:

                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            except requests.exceptions.Timeout:

                last_error = (
                    "Image generation timed out."
                )

            except requests.exceptions.RequestException as e:

                last_error = str(e)

            except Exception as e:

                last_error = str(e)

        return jsonify({
            "ok": False,
            "error": last_error
        }), 502

    except Exception as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# 404
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "ok": False,
        "error": "Endpoint not found"
    }), 404


# --------------------------------------------------
# 500
# --------------------------------------------------

@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "ok": False,
        "error": "Internal server error"
    }), 500


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
                )
