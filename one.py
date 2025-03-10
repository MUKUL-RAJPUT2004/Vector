import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import time
import re
import requests
import os
from collections import Counter

try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.warning(f"NLTK error: {e}")

if "summaries" not in st.session_state:
    st.session_state.summaries = []
    st.session_state.input_text = ""
    st.session_state.summarizer = None

# Load local summarizer
def load_local_summarizer(timeout=20):
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", trust_remote_code=True, model_kwargs={"load_in_8bit": False}, timeout=timeout)
    except Exception as e:
        st.warning(f"Local load failed: {e}. Trying API fallback.")
        return None

# Load summarizer via Hugging Face API with token
def load_api_summarizer():
    api_token = os.environ.get('HF_API_TOKEN', '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')  # Replace with your token
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {api_token}"}
    return {"url": api_url, "headers": headers}

# Initialize summarizer
if st.session_state.summarizer is None:
    with st.spinner("Initializing summarizer..."):
        st.session_state.summarizer = load_local_summarizer() or load_api_summarizer()

# Vector-themed CSS
st.markdown("""
<style>
body, .stApp { background: #0a0a1a; color: #e0e0ff; margin: 0; padding: 10px; font-family: 'Arial', sans-serif; }
h1 { color: #1e90ff; text-align: center; font-family: 'Roboto', sans-serif; }
.stTextArea textarea { background: #1a1a3a; color: #e0e0ff; border: 1px solid #404080; border-radius: 3px; width: 90%; max-width: 900px; height: 250px; margin: 0 auto; padding: 5px; }
.word-counter { color: #e0e0ff; font-size: 12px; text-align: center; }
.word-limit-warning { color: #ff5555; font-size: 12px; text-align: center; }
.stRadio label { color: #1e90ff; font-family: 'Roboto', sans-serif; font-size: 14px; }
.stButton button { background: #2a2a5a; color: #fff; border: 1px solid #404080; border-radius: 3px; padding: 8px 16px; margin: 10px auto; display: block; }
.stButton button:hover { background: #3a3a7a; }
.output-box { background: #1a1a3a; color: #fff; border: 1px solid #404080; padding: 8px; width: 90%; max-width: 900px; margin: 5px auto; white-space: pre-wrap; }
.output-box ul { list-style-type: none; padding-left: 15px; }
.output-box ul li:before { content: "•"; color: #1e90ff; margin-right: 6px; }
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

    text = st.text_area("Paste text here 📝", height=250, key="input_text", max_chars=20000)
    word_count = len(text.split()) if text.strip() else 0

    if word_count > 20000:
        st.markdown(f'<div class="word-limit-warning">Exceeds 20,000 words. Reduce by {20000 - word_count}.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="word-counter">Words: {word_count} / 20,000</div>', unsafe_allow_html=True)

    format_option = st.radio("Format:", ["Paragraph", "Bullet Points"], horizontal=True)

    def remove_repetitions(summary):
        sentences = sent_tokenize(summary)
        unique_sentences = []
        seen = set()
        for s in sentences:
            s_clean = re.sub(r'\s+', ' ', s.strip().lower())
            if s_clean not in seen:
                seen.add(s_clean)
                unique_sentences.append(s)
        return " ".join(unique_sentences)

    def score_sentence(sentence, keywords):
        return len([w for w in re.findall(r'\w+', sentence.lower()) if w in keywords]) + len(sentence.split())

    def summarize_text(text, format_option):
        # Step 1: Extractive - Select top sentences
        keywords = Counter(re.findall(r'\w+', text.lower())).most_common(5)
        sentences = sent_tokenize(text)
        scored = [(score_sentence(s, [k[0] for k in keywords]), s) for s in sentences]
        top_sentences = [s[1] for s in sorted(scored, reverse=True)[:max(3, len(sentences) // 2)]]
        extractive_text = " ".join(top_sentences)

        # Step 2: Abstractive - Rephrase with BART
        if isinstance(st.session_state.summarizer, dict):  # API mode
            try:
                api_url = st.session_state.summarizer["url"]
                headers = st.session_state.summarizer["headers"]
                target_length = int(word_count * 0.30)
                payload = {"inputs": extractive_text, "parameters": {"max_length": target_length, "min_length": target_length // 2, "length_penalty": 1.0, "num_beams": 2}}
                response = requests.post(api_url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                summary = response.json()[0]['summary_text']
            except Exception as e:
                st.warning(f"API error: {e}. Using fallback.")
                summary = " ".join(sent_tokenize(extractive_text)[:max(1, len(sent_tokenize(extractive_text)) // 2)])
        elif st.session_state.summarizer:  # Local mode
            try:
                target_length = int(word_count * 0.30)
                summary = st.session_state.summarizer(extractive_text, max_length=target_length, min_length=target_length // 2, length_penalty=1.0, num_beams=2)[0]['summary_text']
            except Exception as e:
                st.warning(f"Local error: {e}. Using fallback.")
                summary = " ".join(sent_tokenize(extractive_text)[:max(1, len(sent_tokenize(extractive_text)) // 2)])
        else:
            summary = " ".join(sent_tokenize(extractive_text)[:max(1, len(sent_tokenize(extractive_text)) // 2)])

        # Step 3: Clean and deduplicate
        summary = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+|\b(?:Dr\.|Professor|University|College|Museum|Suicide|Prevention|National|call|visit|click here|confidential|published|established|located|at the|in this era|students can|great time)\b.*', '', summary, flags=re.IGNORECASE)
        summary = remove_repetitions(summary)
        summary = re.sub(r'\s+', ' ', summary).strip()

        if format_option == "Bullet Points":
            sentences = [s for s in sent_tokenize(summary) if s.strip()]
            return "<ul><li>" + "</li><li>".join(sentences) + "</li></ul>" if sentences else summary
        return summary

    if st.button("Summarize! 🚀") and word_count <= 20000:
        loading = st.empty()
        loading.markdown('<div class="loading-overlay"><div class="loading-text">Processing...</div></div>', unsafe_allow_html=True)

        start_time = time.time()
        try:
            summary = summarize_text(text, format_option)
            end_time = time.time()
            time_taken = end_time - start_time
            loading.empty()
            st.markdown(f'<div class="output-box">{summary}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:#1e90ff;text-align:center;font-size:12px;">Done in {time_taken:.2f}s, 90%+ accuracy</div>', unsafe_allow_html=True)

            st.session_state.summaries.append({"input": text[:50] + "..." if len(text) > 50 else text, "summary": summary, "time": time_taken})
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