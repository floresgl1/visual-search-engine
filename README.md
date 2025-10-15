# visual-search-engine
smart visual search engine using RAG and computer vision
<<<<<<< HEAD
=======

>>>>>>> 6345e96 (initial project setup with structure and dependencies)
# Visual Search Engine

A smart visual search engine that allows you to search through image collections using natural language queries. Built with PyTorch, TensorFlow, OpenCV, and RAG (Retrieval-Augmented Generation).

## 🎯 Project Overview

This project demonstrates:
- **Computer Vision**: Image preprocessing and feature extraction
- **RAG**: Vector database storage and semantic search
- **Deep Learning**: CLIP model for image-text embeddings
- **MLOps**: Docker containerization and deployment

## 🛠️ Tech Stack

- **PyTorch**: Primary framework for CLIP model
- **TensorFlow**: Alternative implementation
- **OpenCV**: Image preprocessing
- **ChromaDB/FAISS**: Vector database for RAG
- **Docker**: Containerization
- **FastAPI**: REST API backend
- **Google Colab**: Model training and experimentation

## 📁 Project Structure

```
visual-search-engine/
├── data/
│   ├── raw/              # Original images
│   ├── processed/        # Preprocessed images
│   └── embeddings/       # Stored embeddings
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_generation.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   ├── preprocessing/
│   │   └── image_processor.py
│   ├── embeddings/
│   │   ├── pytorch_embedder.py
│   │   └── tensorflow_embedder.py
│   ├── retrieval/
│   │   └── vector_db.py
│   └── api/
│       └── app.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/visual-search-engine.git
cd visual-search-engine
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download sample dataset (instructions in notebooks)

### Usage

1. **Data Preparation**: Run `notebooks/01_data_exploration.ipynb`
2. **Generate Embeddings**: Run `notebooks/02_embedding_generation.ipynb`
3. **Test Search**: Run `notebooks/03_retrieval_experiments.ipynb`
4. **Start API**: `python src/api/app.py`

## 📊 Datasets

- MS COCO (330K images with captions)
- Flickr30k (31K images with descriptions)
- Unsplash Lite (25K+ high-quality photos)

## 🔍 Example Queries

- "red car at sunset"
- "person walking dog in park"
- "modern architecture with glass"
- "beach with palm trees"

## 📈 Project Roadmap

- [x] Phase 1: Setup and data preparation
- [ ] Phase 2: PyTorch embedding generation
- [ ] Phase 3: RAG implementation
- [ ] Phase 4: TensorFlow alternative
- [ ] Phase 5: API development
- [ ] Phase 6: Dockerization
- [ ] Phase 7: Deployment

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI CLIP model
- MS COCO dataset
- ChromaDB team
