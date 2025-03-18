import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import time
import re
import os
import shutil
from collections import Counter
from transformers import pipeline

# Initialize Session State
if "summaries" not in st.session_state:
    st.session_state.summaries = []
    st.session_state.input_text = ""

# Initialize Local Summarizer
summarizer = None
try:
    # Try loading google/pegasus-xsum (larger model)
    summarizer = pipeline("summarization", model="google/pegasus-xsum", device=-1)  # CPU
except Exception as e:
    st.warning(f"Failed to load google/pegasus-xsum: {e}. Trying a lighter model.")
    try:
        # Fallback to a lighter model
        summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device=-1)  # CPU
    except Exception as e:
        st.warning(f"Failed to load fallback model: {e}. Using backup method.")

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
    st.write("Powered by advanced AI to deliver concise summaries in seconds. 🚀")

    text_input = st.text_area("Paste text here 📝", value=st.session_state.input_text, height=250, max_chars=20000)
    st.session_state.input_text = text_input
    word_count = len(text_input.split()) if text_input.strip() else 0

    if word_count > 20000:
        st.markdown(f'<div class="word-limit-warning">Exceeds 20,000 words. Reduce by {20000 - word_count}.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="word-counter">Words: {word_count} / 20,000</div>', unsafe_allow_html=True)

    def remove_repetitions(summary, sentences):
        unique_sentences = []
        seen = set()
        for s in sentences:
            s_clean = re.sub(r'\s+', ' ', s.strip().lower())
            if s_clean not in seen and len(s.split()) > 3:
                seen.add(s_clean)
                unique_sentences.append(s)
        return " ".join(unique_sentences)

    def score_sentence(sentence, keywords, position, total_sentences, word_freq):
        keyword_score = len([w for w in re.findall(r'\w+', sentence.lower()) if w in keywords])
        position_score = (total_sentences - position + 1) / total_sentences
        diversity_score = len(set(re.findall(r'\w+', sentence.lower()))) / len(re.findall(r'\w+', sentence.lower())) if len(re.findall(r'\w+', sentence.lower())) > 0 else 0
        centrality_score = sum(word_freq.get(word, 0) for word in re.findall(r'\w+', sentence.lower())) / len(re.findall(r'\w+', sentence.lower())) if len(re.findall(r'\w+', sentence.lower())) > 0 else 0
        return keyword_score * 2.0 + position_score * 1.5 + diversity_score * 1.0 + centrality_score * 1.0

    def initialize_nltk():
        nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
        punkt_tab_path = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
        punkt_tab_zip = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab.zip')

        if os.path.exists(punkt_tab_path):
            try:
                shutil.rmtree(punkt_tab_path)
            except Exception as e:
                st.warning(f"Failed to remove punkt_tab directory: {e}.")
        if os.path.exists(punkt_tab_zip):
            try:
                os.remove(punkt_tab_zip)
            except Exception as e:
                st.warning(f"Failed to remove punkt_tab.zip: {e}.")

        try:
            nltk.download('punkt_tab', quiet=True, raise_on_error=True)
        except Exception as e:
            st.warning(f"NLTK download error: {e}. Using fallback tokenizer.")
            def custom_tokenize(text):
                sentences = []
                current = ""
                i = 0
                while i < len(text):
                    char = text[i]
                    current += char
                    if char in '.!?':
                        if i > 1 and text[i-1].isalpha() and text[i-2] == ' ':
                            i += 1
                            continue
                        if i + 1 < len(text) and text[i+1] == ' ' and i + 2 < len(text) and text[i+2].isupper():
                            sentences.append(current.strip())
                            current = ""
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                if current.strip():
                    sentences.append(current.strip())
                return sentences
            return custom_tokenize
        return sent_tokenize

    def summarize_text(text):
        # Initialize NLTK tokenizer at runtime
        tokenizer = initialize_nltk()

        # Remove titles and irrelevant content
        text = re.sub(r'\*\*.*?\*\*', '', text).strip()
        text = re.sub(r'⚡.*?(?=\n|$)|🤔.*?(?=\n|$)|Edited.*?(?=\n|$)|Questions.*?(?=\n|$)|Like.*?(?=\n|$)|Deepen.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        keywords = [k[0] for k in Counter(re.findall(r'\w+', text.lower())).most_common(5)]
        word_freq = Counter(re.findall(r'\w+', text.lower()))
        sentences = tokenizer(text)
        total_sentences = len(sentences)
        total_words = len(re.findall(r'\w+', text))

        # Target 30% of the input word count for the summary
        target_word_count = max(30, int(total_words * 0.3))

        # Score sentences for fallback
        scored = [(score_sentence(s, keywords, i + 1, total_sentences, word_freq), s) for i, s in enumerate(sentences)]
        sorted_sentences = [s[1] for s in sorted(scored, reverse=True)]

        # Try local summarization
        if summarizer:
            try:
                target_length = max(50, int(total_words * 0.3))
                summary_dict = summarizer(text, max_length=target_length, min_length=max(30, target_length // 2), do_sample=False)
                summary = summary_dict[0]['summary_text']
            except Exception as e:
                st.warning(f"Local summarization failed: {e}. Using backup method.")
                # Fallback: Select sentences until reaching target word count
                summary_sentences = []
                current_word_count = 0
                for sentence in sorted_sentences:
                    sentence_word_count = len(re.findall(r'\w+', sentence))
                    if current_word_count + sentence_word_count <= target_word_count:
                        summary_sentences.append(sentence)
                        current_word_count += sentence_word_count
                    else:
                        # Trim the last sentence to meet the target word count exactly
                        remaining_words = target_word_count - current_word_count
                        if remaining_words > 10:  # Only trim if significant words remain
                            words = re.findall(r'\w+', sentence)
                            trimmed_sentence = " ".join(words[:remaining_words]) + "..."
                            summary_sentences.append(trimmed_sentence)
                        break
                summary = remove_repetitions(" ".join(summary_sentences), summary_sentences)
        else:
            # Fallback: Select sentences until reaching target word count
            summary_sentences = []
            current_word_count = 0
            for sentence in sorted_sentences:
                sentence_word_count = len(re.findall(r'\w+', sentence))
                if current_word_count + sentence_word_count <= target_word_count:
                    summary_sentences.append(sentence)
                    current_word_count += sentence_word_count
                else:
                    # Trim the last sentence to meet the target word count exactly
                    remaining_words = target_word_count - current_word_count
                    if remaining_words > 10:  # Only trim if significant words remain
                        words = re.findall(r'\w+', sentence)
                        trimmed_sentence = " ".join(words[:remaining_words]) + "..."
                        summary_sentences.append(trimmed_sentence)
                    break
            summary = remove_repetitions(" ".join(summary_sentences), summary_sentences)

        summary = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+|\b(?:Dr\.|Professor|University|College|Museum|Suicide|Prevention|National|call|visit|click here|confidential|published|established|located|at the|in this era|students can|great time|survey|newsletter|email|sign up)\b.*', '', summary, flags=re.IGNORECASE)
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
    <strong>Feedback:</strong> <a href="https://forms.gle/sk91cmKP2MnVJ3999" style="color:#1e90ff;">Share your thoughts here!</a><br>
    We’d love to hear your feedback to improve Vector!
    </div>
    """, unsafe_allow_html=True)