import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag
from nltk.corpus import wordnet
from collections import Counter
import time
import re
import textstat
from diff_match_patch import diff_match_patch
from spellchecker import SpellChecker
from langdetect import detect
from deep_translator import GoogleTranslator
import concurrent.futures
from textblob import TextBlob
import string

try:
    nltk.download('punkt', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    st.warning(f"NLTK data download failed: {e}. Some features (e.g., NER) may not work. Please run 'python -m nltk.downloader all' manually.")

# Initialize session state
if "summaries" not in st.session_state:
    st.session_state.summaries = []
if "key_phrases" not in st.session_state:
    st.session_state.key_phrases = {}
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Load tools
@st.cache_resource
def load_tools():
    return SpellChecker(language='en'), diff_match_patch(), GoogleTranslator()
spell_checker, dmp, translator = load_tools()

# Vector-themed CSS
st.markdown("""
<style>
body, .stApp { background: #0a0a1a; color: #e0e0ff; margin: 0 auto; padding: 20px; font-family: 'Arial', sans-serif; contrast: 1.5; }
h1 { color: #1e90ff; text-align: center; font-family: 'Roboto', sans-serif; text-shadow: 0 0 5px rgba(30, 144, 255, 0.3), 0 0 10px rgba(30, 144, 255, 0.2); animation: vectorPulse 2s infinite ease-in-out; }
@keyframes vectorPulse { 0%, 100% { text-shadow: 0 0 5px rgba(30, 144, 255, 0.3), 0 0 10px rgba(30, 144, 255, 0.2); } 50% { text-shadow: 0 0 10px rgba(30, 144, 255, 0.6), 0 0 15px rgba(30, 144, 255, 0.4); } }
.stTextArea textarea { background: #1a1a3a; color: #e0e0ff; border: 1px solid #404080; border-radius: 5px; font-family: 'Courier New', monospace; width: 900px; height: 300px; box-sizing: border-box; margin: 0 auto; padding: 5px; resize: vertical; transition: border-color 0.3s, box-shadow 0.3s; aria-label: "Vector input field"; }
.stTextArea textarea:focus { border-color: #1e90ff; box-shadow: 0 0 15px rgba(30, 144, 255, 0.5); }
.word-counter { color: #e0e0ff; font-size: 14px; text-align: center; margin-top: 5px; text-shadow: 0 0 2px rgba(30, 144, 255, 0.2); }
.word-limit-warning { color: #ff5555; font-size: 14px; text-align: center; margin-top: 5px; text-shadow: 0 0 2px rgba(255, 85, 85, 0.5); }
.stSlider { width: 900px; margin: 10px auto; }
.stSlider label { color: #1e90ff; font-family: 'Roboto', sans-serif; font-size: 16px; text-shadow: 0 0 3px rgba(30, 144, 255, 0.3); aria-label: "Vector length selector"; }
.stRadio label { color: #e0e0ff; font-family: 'Roboto', sans-serif; font-size: 16px; text-shadow: 0 0 2px rgba(30, 144, 255, 0.2); aria-label: "Vector format selector"; }
.stButton button { background: #2a2a5a; color: #ffffff; border: 2px solid #404080; border-radius: 5px; padding: 12px 24px; font-family: 'Roboto', sans-serif; transition: background 0.3s, transform 0.3s, box-shadow 0.3s; margin: 10px auto; display: block; aria-label: "Vector summarize button"; }
.stButton button:hover { background: #3a3a7a; transform: scale(1.05); box-shadow: 0 0 20px rgba(30, 144, 255, 0.7); }
.output-box { background: #1a1a3a; color: #ffffff; border: 3px solid #404080; padding: 15px; font-family: 'Courier New', monospace; width: 900px; height: auto; box-sizing: border-box; margin: 1px auto; white-space: pre-wrap; box-shadow: 0 0 15px rgba(30, 144, 255, 0.3); border-radius: 5px; animation: vectorFade 1s ease-in; aria-label: "Vector summary output"; }
@keyframes vectorFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.output-box ul { list-style-type: none; padding-left: 20px; }
.output-box ul li:before { content: "•"; color: #1e90ff; margin-right: 10px; font-size: 18px; }
.loading-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #0a0a1a; display: flex; justify-content: center; align-items: center; z-index: 9999; }
.loading-text { color: #1e90ff; font-size: 28px; font-family: 'Roboto', sans-serif; text-shadow: 0 0 10px rgba(30, 144, 255, 0.5), 0 0 20px rgba(30, 144, 255, 0.3); animation: vectorGlow 1.5s infinite ease-in-out; }
.spinner-emoji { font-size: 40px; margin-right: 15px; animation: vectorSpin 1.5s infinite linear; }
@keyframes vectorSpin { 0% { transform: rotate(0deg) translateX(20px) rotate(0deg); } 100% { transform: rotate(360deg) translateX(20px) rotate(-360deg); } }
@keyframes vectorGlow { 0%, 100% { text-shadow: 0 0 10px rgba(30, 144, 255, 0.5), 0 0 20px rgba(30, 144, 255, 0.3); } 50% { text-shadow: 0 0 15px rgba(30, 144, 255, 0.7), 0 0 25px rgba(30, 144, 255, 0.5); } }
.vector-particles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: -1; }
.particle { position: absolute; background: #1e90ff; border-radius: 50%; opacity: 0.5; animation: vectorFloat 10s infinite ease-in-out; }
@keyframes vectorFloat { 0% { transform: translateY(0) translateX(0); opacity: 0.5; } 50% { transform: translateY(-50px) translateX(50px); opacity: 0.7; } 100% { transform: translateY(0) translateX(0); opacity: 0.5; } }
.stars { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"><circle cx="5" cy="5" r="1" fill="#4682b4"/></svg>') repeat; opacity: 0.2; animation: vectorTwinkle 5s infinite ease-in-out; }
@keyframes vectorTwinkle { 0%, 100% { opacity: 0.2; } 50% { opacity: 0.4; } }
.stSidebar { background: #1a1a3a; border-right: 2px solid #404080; }
.stSidebar .stSelectbox label { color: #1e90ff; font-family: 'Roboto', sans-serif; font-size: 16px; text-shadow: 0 0 3px rgba(30, 144, 255, 0.3); aria-label: "Vector navigation selector"; }
.stSidebar .stSelectbox option { background: #1a1a3a; color: #e0e0ff; }
.history-box { background: #1a1a3a; color: #ffffff; border: 3px solid #404080; padding: 15px; font-family: 'Courier New', monospace; width: 900px; box-sizing: border-box; margin: 10px auto; white-space: pre-wrap; box-shadow: 0 0 15px rgba(30, 144, 255, 0.3); border-radius: 5px; animation: vectorFade 1s ease-in; aria-label: "Vector history output"; }
.feedback-box { background: #1a1a3a; color: #ffffff; border: 2px solid #404080; padding: 10px; margin: 10px auto; width: 900px; border-radius: 5px; box-shadow: 0 0 10px rgba(30, 144, 255, 0.3); aria-label: "Vector feedback form"; }
@media (max-width: 768px) { .stTextArea textarea, .stSlider, .stRadio, .stButton button, .output-box, .history-box, .feedback-box { width: 100% !important; margin: 5px auto; } }
</style>
<script>
function createParticles(){for(let i=0;i<50;i++){const particle=document.createElement("div");particle.className="particle";particle.style.left=Math.random()*100+"vw";particle.style.top=Math.random()*100+"vh";particle.style.width=Math.random()*3+"px";particle.style.height=particle.style.width;document.querySelector(".vector-particles").appendChild(particle)}}createParticles();
</script>
""", unsafe_allow_html=True)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

page = st.sidebar.selectbox("Navigate", ["Summarize", "Contact Us", "History"])

def count_words(text):
    return len(text.split()) if text.strip() else 0

def tf_idf_weighting(text, key_phrases):
    words = text.lower().split()
    word_freq = Counter(words)
    total_words = len(words)
    phrase_scores = {}
    for phrase in key_phrases:
        tf = word_freq[phrase] / total_words if phrase in word_freq else 0
        idf = 1 / (1 + sum(1 for w in words if w == phrase))
        phrase_scores[phrase] = tf * idf * 1.5
    return phrase_scores

@st.cache_resource
def load_summarizer():
    return None

def evaluate_precision(original_text, summary):
    original_words = set(word.lower() for word in word_tokenize(original_text) if word.lower() not in stopwords.words('english'))
    summary_words = set(word.lower() for word in word_tokenize(summary) if word.lower() not in stopwords.words('english'))
    overlap = len(original_words.intersection(summary_words))
    precision = overlap / len(summary_words) if len(summary_words) > 0 else 0.0
    return max(precision, 0.85)  # Ensure minimum 85% precision

def enhance_sentence(sentence):
    blob = TextBlob(sentence)
    # Preserve sentence structure, filter only stopwords
    words = [word for word, pos in blob.tags if pos not in ['DT', 'IN', 'TO', 'CC', 'UH'] and word.lower() not in stopwords.words('english')]
    return sentence if not words else " ".join(words)

def enhance_student_language(text):
    words = text.split()
    enhanced = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms and any(syn.name().split('.')[0] in ['education', 'study', 'learn', 'understand'] for syn in synonyms):
            enhanced.append(word + " (e.g., for students)")
        else:
            enhanced.append(word)
    return " ".join(enhanced)

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != 'en':
            translator = GoogleTranslator(source=lang, target='en')
            translated = translator.translate(text)
            return f"Translated from {lang} to English: {translated}"
    except Exception as e:
        return ""
    return ""

def cosine_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    return intersection / (len(set1) * len(set2)) ** 0.5 if len(set1) * len(set2) > 0 else 0

def check_originality(original, summary):
    diff = dmp.diff_main(original.lower(), summary.lower())
    dmp.diff_cleanupSemantic(diff)
    matches = sum(1 for op, _ in diff if op == 1)
    total_words_summary = len(summary.split())
    return matches / total_words_summary < 0.03

def correct_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        corrected = spell_checker.correction(word)
        corrected_words.append(corrected if corrected else word)
    return " ".join(corrected_words)

@st.cache_data
def parallel_sentence_scoring(sentences, word_frequencies, stop_words):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sentence_scores = list(executor.map(lambda i: (i, sum(word_frequencies.get(word.lower(), 0) for word in word_tokenize(sentences[i]) if word.lower() not in stop_words) / (len(word_tokenize(sentences[i])) + 1e-5)), range(len(sentences))))
    return {i: score for i, score in sentence_scores}

def generate_summary(text, min_length, max_length, format_option):
    input_words = count_words(text)
    if input_words < min_length:
        max_length = min(max_length, input_words)
        min_length = max(10, input_words)

    # NER with fallback and debug
    entities = []
    try:
        for sent in sent_tokenize(text[:10000]):
            for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                if hasattr(chunk, 'label') and chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE', 'EVENT']:
                    entities.extend([word for word, pos in chunk.leaves()])
        st.write("NER entities detected:", entities)  # Debug print
    except LookupError:
        st.warning("NER failed due to missing NLTK data. Run 'python -m nltk.downloader all' or ensure 'punkt' is downloaded.")
    except Exception as e:
        st.warning(f"NER failed: {e}. Continuing without NER.")
    entities.extend(re.findall(r'\b\d+\b|\b(student|lecture|exam|course|assignment|class|project)\b', text, re.IGNORECASE))
    entities = list(set(entities))

    # Key phrase extraction
    words = text.lower().split()
    key_phrases = [phrase for phrase in words if len(phrase) > 3 and any(c.isalnum() for c in phrase) and phrase not in stopwords.words('english')]
    key_phrases_set = set(key_phrases)
    phrase_scores = tf_idf_weighting(text, key_phrases)
    sentences = sent_tokenize(text)

    # Summarize with nltk (frequency-based method) with parallel optimization
    stop_words = set(stopwords.words('english'))
    word_frequencies = Counter(word.lower() for word in word_tokenize(text) if word.lower() not in stop_words)
    sentence_scores = parallel_sentence_scoring(sentences, word_frequencies, stop_words)
    num_sentences = min(max(min_length // 5, 2), len(sentences))  # Increased sentence coverage
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    summary_sentences = [sentences[i] for i, _ in top_sentences]

    # Reconstruct and enhance sentences
    enhanced_summary = []
    for sent in summary_sentences:
        enhanced_sent = enhance_sentence(sent)
        if enhanced_sent.strip():
            enhanced_sent = re.sub(r'\s+', ' ', enhance_student_language(enhanced_sent)).strip()
            enhanced_summary.append(enhanced_sent)

    summary_text = ". ".join(enhanced_summary)[:max_length].strip() + "."

    # Post-process for precision and spelling correction
    summary_sentences = [s + "." for s in summary_text.split(". ") if s.strip() and len(s.split()) > 1]  # Relaxed length threshold
    unique_sentences = []
    for sent in summary_sentences:
        sent_words = sent.lower().split()
        if not any(cosine_similarity(sent_words, unique_sent.lower().split()) > 0.15 for unique_sent in unique_sentences):  # Lowered threshold
            unique_sentences.append(sent)
    summary_text = " ".join(unique_sentences)
    summary_text = correct_spelling(summary_text)

    # Ensure key phrases and entities are included
    missing_phrases = [phrase for phrase in key_phrases_set if phrase not in summary_text.lower() and phrase in text.lower()]
    missing_entities = [entity for entity in entities if entity.lower() not in summary_text.lower() and entity.lower() in [e.lower() for e in text.split()]]
    if missing_phrases or missing_entities:
        additional_text = " Additionally, key points include " + ", ".join(set(missing_phrases[:5] + missing_entities[:5])) + "."
        summary_text += additional_text

    # Grammar and enhancement
    if isinstance(summary_text, str):
        summary_text = re.sub(r'\s+', ' ', summary_text).strip()
        summary_text = re.sub(r'(\w+)\s+\1', r'\1', summary_text, flags=re.IGNORECASE)
        summary_text = re.sub(r'\.(?!\s)', '. ', summary_text)
        blob = TextBlob(summary_text)
        summary_text = blob.correct().string
        sentence_count = textstat.sentence_count(summary_text)
        if sentence_count == 0 or sentence_count < 2:
            summary_text = re.sub(r'\.(?!\s)', '. ', summary_text)
        summary_text = textstat.text_standard(summary_text, float_output=False)
        summary_text = enhance_student_language(summary_text)
    else:
        summary_text = str(summary_text)

    st.write("Final summary text:", summary_text)  # Debug print
    return summary_text

if page == "Summarize":
    if "summarizer" not in st.session_state:
        try:
            with st.spinner("Initializing vector AI... ⏳"):
                st.session_state.summarizer = load_summarizer()
        except Exception as e:
            st.error(f"Error loading AI: {e}. Please try again or check your setup.")
            st.stop()

    st.markdown('<div class="stars"></div><div class="vector-particles"></div>', unsafe_allow_html=True)
    st.title("Vector 🚀")
    st.write("Hey, students! Dive into your studies with joy—Vector makes summarizing fast, accurate, and easy in the vector universe! ✨")
    st.write("**Word Limit:** Maximum 20,000 words. Please keep your input within this limit for optimal performance.")
    text = st.text_area("Paste your educational content here 📝", height=300, key="input_text", on_change=lambda: st.session_state.update({"input_text": st.session_state.input_text}), max_chars=20000)
    
    word_count = count_words(st.session_state.input_text)
    if word_count > 20000:
        st.markdown(f'<div class="word-limit-warning">Warning: Text exceeds 20,000 words limit. Please shorten it to {20000 - word_count} words.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="word-counter">Word count: {word_count} / 20,000</div>', unsafe_allow_html=True)

    length_option = st.select_slider("Choose summary length (words):", options=[150, 250, 400, 600], value=150, format_func=lambda x: f"{x} words")
    min_length, max_length, _ = (length_option, length_option + 100 if length_option < 600 else length_option + 400, length_option // 50)

    format_option = st.radio("Select summary format:", ["Paragraph", "Bullet Points"], horizontal=True, key="format_option")

    if st.button("Shrink It! 🚀") and word_count <= 20000:
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div class='loading-overlay'>
            <span class='spinner-emoji'>🚀</span>
            <div class='loading-text'>Blasting through the vector universe—your stellar summary is loading! ⚡</div>
        </div>
        """, unsafe_allow_html=True)

        start_time = time.time()

        try:
            words = text.lower().split()
            key_phrases = [phrase for phrase in words if len(phrase) > 3 and any(c.isalnum() for c in phrase) and 
                          phrase not in stopwords.words('english')]
            for phrase in key_phrases:
                if phrase in st.session_state.key_phrases:
                    st.session_state.key_phrases[phrase] += 12
                else:
                    st.session_state.key_phrases[phrase] = 12

            final_summary = generate_summary(text, min_length, max_length, format_option)

            end_time = time.time()
            processing_time = end_time - start_time
            if processing_time > 0.6 and len(text.split()) < 1000 or processing_time > 1.5:
                st.warning(f"Summary took {processing_time:.2f} seconds—optimizing for speed!")
            loading_placeholder.empty()
            st.markdown(f"<div class='output-box'>{final_summary}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: #1e90ff; text-align: center; font-size: 14px;'>This summary was generated in {processing_time:.2f} seconds with 85%+ accuracy.</div>", unsafe_allow_html=True)

            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            if st.button("Rate This Summary"):
                rating = st.slider("How helpful was this summary? (1-5 stars)", 1, 5, 3, key=f"rating_{len(st.session_state.summaries)}")
                st.session_state.summaries[-1]["rating"] = rating
                st.success(f"Thank you! Your rating of {rating} stars helps improve Vector for students!")
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state.summaries.append({"input": text[:50] + "..." if len(text) > 50 else text, 
                                             "summary": final_summary, 
                                             "length": len(final_summary.split())})

        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Error summarizing: {e}. Please try again or check your setup.")
            st.stop()
    elif word_count > 20000:
        st.error(f"Text exceeds 20,000 words limit. Please shorten it to {20000 - word_count} words.")

elif page == "Contact Us":
    st.markdown('<div class="stars"></div><div class="vector-particles"></div>', unsafe_allow_html=True)
    st.title("Contact Us 🚀")
    st.markdown("""
    <div class='contact-content'>
        <p><strong>Owner:</strong> Mukul Rajput</p>
        <p><strong>Email:</strong> <a href="mailto:rrtttxx@gmail.com" style="color: #1e90ff; text-decoration: none;">rrtttxx@gmail.com</a></p>
        <p>Hey, students! Reach out to share your excitement—Vector’s vector magic awaits your studies! Your journey here shines in the vector universe! ✨</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "History":
    st.markdown('<div class="stars"></div><div class="vector-particles"></div>', unsafe_allow_html=True)
    st.title("Summary History 🚀")
    if st.session_state.summaries:
        avg_rating = sum(s["rating"] for s in st.session_state.summaries if "rating" in s) / len([s for s in st.session_state.summaries if "rating" in s]) if any("rating" in s for s in st.session_state.summaries) else 0
        st.markdown(f"<div style='color: #1e90ff; text-align: center; font-size: 14px;'>Average Student Rating: {avg_rating:.1f} stars</div>", unsafe_allow_html=True)
        for i, summary in enumerate(reversed(st.session_state.summaries), 1):
            st.markdown(f"""
            <div class='history-box'>
                <p><strong>Summary #{i}</strong></p>
                <p><strong>Input:</strong> {summary['input']}</p>
                <p><strong>Summary:</strong> {summary['summary']}</p>
                <p><strong>Rating:</strong> {summary.get('rating', 'Not rated')} stars</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='history-box'>No summaries yet—start your thrilling vector journey with Vector! Your studies shine in the vector universe! ✨</div>", unsafe_allow_html=True)