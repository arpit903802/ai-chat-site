import os
import time
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Keys Render Environment Variables se aayengi
API_KEYS = [
    os.getenv("NVIDIA_API_KEY_1"),
    os.getenv("NVIDIA_API_KEY_2"),
    os.getenv("NVIDIA_API_KEY_3"),
    os.getenv("NVIDIA_API_KEY_4"),
    os.getenv("NVIDIA_API_KEY_5"),
    os.getenv("NVIDIA_API_KEY_6"),
]

API_KEYS = [k for k in API_KEYS if k]

# Tumhare testing results ke basis par reliable fallback models
TEXT_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-20b",
    "mistralai/mistral-nemotron",
    "meta/muse-glimmer-30b",
    "moonshotai/kimi-k3",
]

VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

# Key rotation
key_index = 0


def get_key():
    global key_index

    if not API_KEYS:
        raise RuntimeError("No NVIDIA API keys configured.")

    key = API_KEYS[key_index % len(API_KEYS)]
    key_index += 1
    return key


def headers(key):
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def ask_nvidia(messages, model=None, max_tokens=1200):
    """
    Multiple keys + multiple models fallback.
    Agar ek key/model fail ho jaye to next combination try karega.
    """

    models = [model] if model else TEXT_MODELS

    last_error = "Unknown error"

    # Har model ko available keys ke saath try karo
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
                    NVIDIA_URL,
                    headers=headers(key),
                    json=payload,
                    timeout=45,
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    if answer:
                        return {
                            "ok": True,
                            "answer": answer,
                            "model": current_model,
                        }

                last_error = (
                    f"{response.status_code}: "
                    f"{response.text[:300]}"
                )

            except requests.exceptions.Timeout:
                last_error = "Request timeout"

            except requests.exceptions.RequestException as e:
                last_error = str(e)

    return {
        "ok": False,
        "error": last_error,
    }


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "keys_configured": len(API_KEYS),
        "models": TEXT_MODELS,
    })


@app.route("/api/chat", methods=["POST"])
def chat():

    try:
        data = request.get_json(force=True)

        messages = data.get("messages", [])

        if not messages:
            return jsonify({
                "error": "No messages provided"
            }), 400

        # Basic limit
        messages = messages[-20:]

        result = ask_nvidia(messages)

        if not result["ok"]:
            return jsonify(result), 502

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/api/vision", methods=["POST"])
def vision():

    try:
        data = request.get_json(force=True)

        image = data.get("image")
        prompt = data.get(
            "prompt",
            "Describe this image accurately."
        )

        if not image:
            return jsonify({
                "error": "Image is required"
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

        if not result["ok"]:
            return jsonify(result), 502

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/api/image", methods=["POST"])
def image_generation():

    """
    Image generation endpoint.

    NVIDIA image-generation model access account/project
    ke according available ho sakta hai.
    """

    try:
        data = request.get_json(force=True)

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Prompt is required"
            }), 400

        # NVIDIA image generation endpoint
        url = (
            "https://ai.api.nvidia.com/v1/genai/"
            "black-forest-labs/flux.1-schnell"
        )

        payload = {
            "prompt": prompt,
            "steps": 4,
            "width": 1024,
            "height": 1024,
            "seed": 0,
        }

        last_error = "Unknown error"

        for _ in range(len(API_KEYS)):

            key = get_key()

            try:
                response = requests.post(
                    url,
                    headers=headers(key),
                    json=payload,
                    timeout=90,
                )

                if response.status_code == 200:

                    data = response.json()

                    # NVIDIA response format
                    image_b64 = data.get("image")

                    if image_b64:
                        return jsonify({
                            "ok": True,
                            "image": (
                                "data:image/png;base64,"
                                + image_b64
                            )
                        })

                    # Alternate response formats
                    if "artifacts" in data:
                        artifacts = data["artifacts"]

                        if artifacts:
                            b64 = artifacts[0].get("base64")

                            if b64:
                                return jsonify({
                                    "ok": True,
                                    "image":
                                        "data:image/png;base64,"
                                        + b64
                                })

                last_error = (
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            except requests.exceptions.Timeout:
                last_error = "Image generation timeout"

            except requests.exceptions.RequestException as e:
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


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found"
    }), 404


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
      )
