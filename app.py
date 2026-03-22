from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

# Environment Variables থেকে কীগুলো নেওয়া
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# এপিআই কনফিগারেশন
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Gemini মডেল সেটিংস (লাইভ সার্চ টুলসহ)
gemini_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[{"google_search_retrieval": {}}]
)

@app.route('/')
def home():
    return "Dibakar AI Hybrid Engine is Live!"

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "No query found"}), 400

    try:
        # ধাপ ১: Llama 3.1 8B থেকে ড্রাফট নেওয়া (Max Tokens: 150)
        llama_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": f"Provide a technical draft for: {user_query}"}],
            model="llama-3.1-8b-instant",
            max_tokens=150
        )
        llama_draft = llama_response.choices[0].message.content

        # ধাপ ২: Gemini-কে ইনস্ট্রাকশন দেওয়া (Live Data + Knowledge Blend)
        # এখানে ভাষার কথা বলা হয়েছে যাতে ইউজার যে ভাষায় প্রশ্ন করবে সেভাবেই উত্তর আসে
        prompt = (
            f"User Question: {user_query}\n"
            f"Llama's Knowledge: {llama_draft}\n\n"
            "Instructions:\n"
            "1. Use your internal knowledge and Llama's draft to understand the topic.\n"
            "2. Use GOOGLE SEARCH to get the latest live data and facts.\n"
            "3. Answer in the SAME LANGUAGE as the User's Question.\n"
            "4. Make the language VERY EASY and simple to understand.\n"
            "5. Keep the total answer under 300 words.\n"
            "6. At the end, provide a 'Sources' section with website links used for search."
        )
        
        gemini_output = gemini_model.generate_content(prompt)
        
        # সোর্স এবং টেক্সট আলাদা করা
        final_answer = gemini_output.text
        
        return jsonify({
            "answer": final_answer,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
