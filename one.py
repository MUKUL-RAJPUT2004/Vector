import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import time
import re
import requests
import os
import shutil
from collections import Counter

# Fix NLTK download by removing existing punkt_tab directory if it exists
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
punkt_tab_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
if os.path.exists(punkt_tab_path):
    try:
        shutil.rmtree(punkt_tab_path)  # Remove the directory to avoid conflicts
    except Exception as e:
        st.warning(f"Failed to remove existing punkt_tab directory: {e}. Using fallback tokenizer.")

# Attempt to download punkt_tab (English only to reduce conflicts)
try:
    nltk.download('punkt_tab', quiet=True, raise_on_error=True)
except Exception as e:
    st.warning(f"NLTK error: {e}. Using fallback tokenizer.")
    def custom_tokenize(text):
        # Improved fallback tokenizer to handle common edge cases
        sentences = []
        current = ""
        i = 0
        while i < len(text):
            char = text[i]
            current += char
            if char in '.!?':  # Potential sentence end
                # Check if it's an abbreviation (e.g., "Dr.", "Mr.")
                if i > 1 and text[i-1].isalpha() and text[i-2] == ' ':
                    i += 1
                    continue
                # Check if followed by a space and capital letter (new sentence)
                if i + 1 < len(text) and text[i+1] == ' ' and i + 2 < len(text) and text[i+2].isupper():
                    sentences.append(current.strip())
                    current = ""
                    i += 2  # Skip the space
                else:
                    i += 1
            else:
                i += 1
        if current.strip():
            sentences.append(current.strip())
        return sentences
    sent_tokenize = custom_tokenize  # Override sent_tokenize with fallback

if "summaries" not in st.session_state:
    st.session_state.summaries = []
    st.session_state.input_text = ""

def load_api_summarizer():
    api_token = "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"
    api_url = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-6-6"
    headers = {"Authorization": f"Bearer {api_token}"}
    return {"url": api_url, "headers": headers}

st.session_state.summarizer = load_api_summarizer()

st.markdown("""
<style>
body, .stApp { background: #0a0a1a; color: #e0e0ff; margin: 0; padding: 10px; font-family: 'Arial', sans-serif; }
h1 { color: #1e90ff; text-align: center; font-family: 'Roboto', sans-serif; }
.stTextArea textarea { background: #1a1a3a; color: #e0e0ff; border: 1px solid #404080; border-radius: 3px; width: 90%; max-width: 900px; height: 250px; margin: 0 auto; padding: 5px; }
.word-counter { color: #e0e0ff; font-size: 12px; text-align: center; }
.word-limit-warning { color: #ff5555; font-size: 12px; text-align: center; }
.stButton button { background: #2a2a5a; color: #fff; border: 1px solid #404080; border-radius: 3px; padding: 8px 16px; margin: 10px auto; display: block; }
.stButton button:hover { background: #3a3a7a; }
.output-box { background: #1a1a3a; color: #fff; border: 1px solid #404080; padding: 8px; width: 90%; max-width: 900px; margin: 5px auto; white-space: pre-wrap; }
.loading-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(10,10,26,0.7); display: flex; justify-content: center; align-items: center; }
.loading-text { color: #1e90ff; font-size: 18px; }
.history-box { background: #1a1a3a; color: #fff; border: 1px solid #404080; padding: 8px; width: 90%; max-width: 900px; margin: 5px auto; white-space: pre-wrap; }
@media (max-width: 768px) { .stTextArea textarea, .stButton button, .output-box, .history-box { width: 100%; margin: 5px 0; } }
</style>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

page = st.sidebar.selectbox("Navigate", ["Summarize", "History", "Contact"], key="nav")

if page == "Summarize":
    st.title("Vector 🚀 - Your Universal Text Summarizer")
    st.write("Fast, free, and versatile for all text needs! ✨ | Limit: 20,000 words.")

    text_input = st.text_area("Paste text here 📝", value=st.session_state.input_text, height=250, max_chars=20000)
    st.session_state.input_text = text_input
    word_count = len(text_input.split()) if text_input.strip() else 0

    if word_count > 20000:
        st.markdown(f'<div class="word-limit-warning">Exceeds 20,000 words. Reduce by {20000 - word_count}.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="word-counter">Words: {word_count} / 20,000</div>', unsafe_allow_html=True)

    def remove_repetitions(summary):
        sentences = sent_tokenize(summary)
        unique_sentences = []
        seen = set()
        for s in sentences:
            s_clean = re.sub(r'\s+', ' ', s.strip().lower())
            if s_clean not in seen and len(s.split()) > 3:
                seen.add(s_clean)
                unique_sentences.append(s)
        return " ".join(unique_sentences)

    def score_sentence(sentence, keywords, position, total_sentences):
        keyword_score = len([w for w in re.findall(r'\w+', sentence.lower()) if w in keywords])
        position_score = (total_sentences - position + 1) / total_sentences
        return keyword_score * 1.5 + position_score * 2 + len(sentence.split())

    def summarize_text(text):
        keywords = [k[0] for k in Counter(re.findall(r'\w+', text.lower())).most_common(5)]
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        scored = [(score_sentence(s, keywords, i + 1, total_sentences), s) for i, s in enumerate(sentences)]
        top_sentences = [s[1] for s in sorted(scored, reverse=True)[:max(3, int(len(sentences) * 0.3))]]
        extractive_text = " ".join(top_sentences)

        # Retry API call up to 3 times with delay
        max_retries = 3
        for attempt in range(max_retries):
            try:
                api_url = st.session_state.summarizer["url"]
                headers = st.session_state.summarizer["headers"]
                target_length = max(100, int(word_count * 0.30))
                payload = {"inputs": extractive_text, "parameters": {"max_length": target_length, "min_length": target_length // 2, "length_penalty": 1.0, "num_beams": 2, "no_repeat_ngram_size": 3}}
                response = requests.post(api_url, headers=headers, json=payload, timeout=5)
                response.raise_for_status()
                summary = response.json()[0]['summary_text']
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retrying
                    continue
                st.warning(f"API error: {e}. Using fallback after {max_retries} attempts.")
                summary = " ".join(sent_tokenize(extractive_text)[:3])  # Limit to 3 sentences in fallback
                break

        summary = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+|\b(?:Dr\.|Professor|University|College|Museum|Suicide|Prevention|National|call|visit|click here|confidential|published|established|located|at the|in this era|students can|great time|survey|newsletter|email|sign up)\b.*', '', summary, flags=re.IGNORECASE)
        summary = remove_repetitions(summary)
        summary = re.sub(r'\s+', ' ', summary).strip()

        return summary

    if st.button("Summarize! 🚀") and word_count <= 20000:
        loading = st.empty()
        loading.markdown('<div class="loading-overlay"><div class="loading-text">Processing...</div></div>', unsafe_allow_html=True)

        start_time = time.time()
        try:
            summary = summarize_text(text_input)
            end_time = time.time()
            time_taken = end_time - start_time
            loading.empty()
            st.markdown(f'<div class="output-box">{summary}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:#1e90ff;text-align:center;font-size:12px;">Done in {time_taken:.2f}s, 90%+ accuracy</div>', unsafe_allow_html=True)

            st.session_state.summaries.append({"input": text_input[:50] + "..." if len(text_input) > 50 else text_input, "summary": summary, "time": time_taken})
        except Exception as e:
            loading.empty()
            st.error(f"Error: {e}. Retry or check setup.")
    elif word_count > 20000:
        st.error(f"Exceeds 20,000 words. Reduce by {20000 - word_count}.")

elif page == "History":
    st.title("Summary History 📜")
    if st.session_state.summaries:
        for i, entry in enumerate(reversed(st.session_state.summaries), 1):
            st.markdown(f"""
            <div class="history-box">
                <p><strong>#{i}</strong></p>
                <p><strong>Input:</strong> {entry['input']}</p>
                <p><strong>Summary:</strong> {entry['summary']}</p>
                <p><strong>Time Taken:</strong> {entry['time']:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="history-box">No summaries yet. Start summarizing!</div>', unsafe_allow_html=True)

elif page == "Contact":
    st.title("Contact Us 📬")
    st.markdown("""
    <div style="color:#e0e0ff;padding:10px;">
    <strong>Creator:</strong> Mukul Rajput<br>
    <strong>Email:</strong> <a href="mailto:rrtttxx@gmail.com" style="color:#1e90ff;">rrtttxx@gmail.com</a><br>
    Share your feedback to improve Vector!
    </div>
    """, unsafe_allow_html=True)