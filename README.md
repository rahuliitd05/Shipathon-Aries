Hosted Project Link - https://huggingface.co/spaces/rahuldoesgithub/factcheck-ai

---
title: FactCheck AI
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🔍 FactCheck AI

An advanced AI-powered fact-checking system that verifies claims using **Natural Language Inference (NLI)**, **semantic similarity**, and **multi-source evidence gathering**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.35+-yellow.svg)

---

## ✨ Features

- **🧠 NLI-Based Verification**: Uses RoBERTa-large-MNLI for entailment/contradiction detection
- **📊 Semantic Similarity**: Sentence-BERT embeddings for evidence matching
- **🌐 Multi-Source Evidence**: Wikipedia, DuckDuckGo, Google Fact Check API
- **⚖️ Ensemble Scoring**: Combines NLI + semantic scores for robust verdicts
- **🎯 Claim Decomposition**: Breaks complex claims into verifiable sub-claims
- **📈 Confidence Scores**: Transparent scoring with detailed breakdowns
- **🎨 Beautiful UI**: Modern Streamlit interface with intuitive design

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Claim Input                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Entity Extraction (spaCy)                     │
│         Extracts: PERSON, ORG, GPE, PRODUCT, EVENT, etc.        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Multi-Source Evidence Gathering                    │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐    │
│  │Wikipedia │  │ DuckDuckGo   │  │ Google Fact Check API  │    │
│  │   API    │  │   Search     │  │                        │    │
│  └──────────┘  └──────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evidence Aggregation                          │
│         Deduplication + Reliability Scoring + Ranking            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NLI Analysis Pipeline                       │
│  ┌─────────────────────┐    ┌─────────────────────────────┐    │
│  │  Semantic Matching  │    │   RoBERTa-large-MNLI        │    │
│  │  (all-MiniLM-L6-v2) │    │   Entailment/Contradiction  │    │
│  └─────────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Verdict Classification                        │
│     TRUE | MOSTLY TRUE | MIXED | MOSTLY FALSE | FALSE           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- ~2GB disk space (for models)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/factcheck-ai.git
   cd factcheck-ai
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Load Models

Click the **"🚀 Load Models"** button to initialize the AI models. This may take 1-2 minutes on first run as models are downloaded (~500MB).

### Step 2: Enter a Claim

Type any factual claim you want to verify in the text box, for example:

- "Barack Obama was born in Hawaii"
- "The Great Wall of China is visible from space"
- "Python was created by Guido van Rossum"

### Step 3: Verify

Click **"🔎 Verify Claim"** and wait for the analysis (typically 10-30 seconds).

### Step 4: Review Results

The system will display:

- **Verdict**: TRUE, MOSTLY TRUE, MIXED, MOSTLY FALSE, FALSE, or UNVERIFIABLE
- **Confidence Score**: How confident the system is in its verdict
- **NLI Score**: Entailment vs contradiction balance
- **Supporting Evidence**: Evidence that supports the claim
- **Contradicting Evidence**: Evidence that contradicts the claim

---

## 🛠️ Project Structure

```
factcheck-ai/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── src/
    ├── __init__.py
    ├── analyzer.py       # NLI + Semantic analysis engine
    └── scraper.py        # Multi-source evidence scraper
```

### Key Components

| File              | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `app.py`          | Streamlit UI with interactive claim verification                |
| `src/analyzer.py` | Ensemble fact-checker with RoBERTa-MNLI + Sentence-BERT         |
| `src/scraper.py`  | Multi-source scraper (Wikipedia, DuckDuckGo, Google Fact Check) |

---

## 🤖 Models Used

| Model                      | Purpose                        | Size   |
| -------------------------- | ------------------------------ | ------ |
| `roberta-large-mnli`       | Natural Language Inference     | ~1.4GB |
| `facebook/bart-large-mnli` | Zero-shot classification       | ~1.6GB |
| `all-MiniLM-L6-v2`         | Semantic similarity embeddings | ~90MB  |
| `en_core_web_sm`           | Named Entity Recognition       | ~12MB  |

---

## 📊 Verdict Definitions

| Verdict             | Meaning                               | Criteria                                 |
| ------------------- | ------------------------------------- | ---------------------------------------- |
| ✅ **TRUE**         | Claim is accurate                     | High entailment, strong evidence support |
| 🟢 **MOSTLY TRUE**  | Largely accurate with minor issues    | Moderate-high entailment                 |
| 🟡 **MIXED**        | Contains both true and false elements | Conflicting evidence                     |
| 🟠 **MOSTLY FALSE** | Largely inaccurate                    | Moderate contradiction                   |
| ❌ **FALSE**        | Claim is inaccurate                   | High contradiction scores                |
| ❓ **UNVERIFIABLE** | Cannot be verified                    | Insufficient evidence                    |

---

## ⚙️ Configuration Options

In the sidebar, you can configure:

- **🌐 Use all sources**: Enable/disable DuckDuckGo and Google Fact Check API
- **🔧 Show debug info**: Display detailed analysis information

---

## 🔧 Technical Details

### NLI Scoring

The system uses a two-stage NLI approach:

1. **Semantic filtering**: Top-5 most similar evidence pieces selected using cosine similarity
2. **NLI classification**: Each evidence piece classified as entailment/neutral/contradiction

### Evidence Reliability Scoring

Sources are weighted by reliability:

- Wikipedia: 0.85
- Reuters/AP News: 0.90
- Snopes/PolitiFact: 0.88
- Google Fact Check: 0.95
- General web sources: 0.50-0.60

### Ensemble Formula

```
support_score = (entailment * 0.7 + semantic_similarity * 0.3) * evidence_confidence
oppose_score = contradiction * evidence_confidence
net_score = support_score - oppose_score
```

---

## 📋 Requirements

```
streamlit>=1.28.0
requests>=2.31.0
beautifulsoup4>=4.12.0
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
spacy>=3.7.0
duckduckgo-search>=4.0.0
numpy>=1.24.0
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"

```bash
pip install -r requirements.txt
```

### "spaCy model not found"

```bash
python -m spacy download en_core_web_sm
```

### Models loading slowly

- First run downloads ~500MB of models
- Subsequent runs use cached models
- Consider using GPU for faster inference

### Out of memory errors

- Close other applications
- Use a machine with 8GB+ RAM
- Set `use_all_sources = False` to reduce memory usage

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for NLI models
- [Sentence-Transformers](https://www.sbert.net/) for semantic embeddings
- [spaCy](https://spacy.io/) for NLP processing
- [Streamlit](https://streamlit.io/) for the web framework
- [DuckDuckGo](https://duckduckgo.com/) for search API

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🚀 Deployment

### Option 1: Hugging Face Spaces (Recommended for ML apps)

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Create a new Space: https://huggingface.co/new-space
3. Select **Streamlit** as the SDK
4. Clone your space and copy project files:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/factcheck-ai
   cd factcheck-ai
   # Copy all project files here
   ```
5. Add the HF Space header to your README.md:
   ```yaml
   ---
   title: FactCheck AI
   emoji: 🔍
   colorFrom: blue
   colorTo: green
   sdk: streamlit
   sdk_version: 1.28.0
   app_file: app.py
   pinned: false
   ---
   ```
6. Push to deploy:
   ```bash
   git add .
   git commit -m "Deploy FactCheck AI"
   git push
   ```

### Option 2: Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and `app.py`
5. Click "Deploy"

> ⚠️ Note: Free tier has 1GB RAM limit. Large models may cause memory issues.

### Option 3: Railway

1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add a `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Deploy

---

<p align="center">
  Made with ❤️ for fighting misinformation
</p>
