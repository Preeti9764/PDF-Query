# PDF Query App

A Streamlit application that allows you to chat with PDF documents using Google's Gemini AI. Upload PDF files, process them, and ask questions about their content.

## Features

- 📄 Upload and process multiple PDF files
- 🤖 Chat with PDFs using Google Gemini AI
- 🔍 Semantic search using FAISS vector store
- 💬 Natural language question answering

## Prerequisites

- Python 3.8 or higher
- Google Generative AI API Key ([Get it here](https://makersuite.google.com/app/apikey))

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd PDFQuery
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   
   Open your browser and navigate to: `http://localhost:8501`

## Usage

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Process**: Click "Submit & Process" to extract and process the text
3. **Ask Questions**: Type your question in the text input and get AI-powered answers

## Project Structure

```
PDFQuery/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── DEPLOYMENT.md       # Detailed deployment guide
├── Dockerfile          # Docker configuration
├── Procfile           # Heroku/Railway configuration
├── render.yaml        # Render.com configuration
└── .gitignore         # Git ignore rules
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment instructions for:

- 🌟 Streamlit Cloud (Recommended)
- 🚀 Render
- 🚊 Railway
- 🐳 Docker
- ☁️ Heroku
- 🔧 AWS EC2

**Quick Deploy (Streamlit Cloud)**:
1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy from GitHub repo
4. Add `GOOGLE_API_KEY` in secrets
5. Done! 🎉

## Technologies Used

- **Streamlit**: Web application framework
- **Google Gemini AI**: LLM for question answering
- **LangChain**: Framework for LLM applications
- **FAISS**: Vector database for semantic search
- **PyPDF2**: PDF text extraction

## Notes

- The FAISS index is stored locally and will be recreated on each deployment (ephemeral storage on most cloud platforms)
- For production use, consider using persistent cloud storage or vector databases like Pinecone or Weaviate
- Make sure to keep your API key secure and never commit it to version control

## License

This project is open source and available under the MIT License.




