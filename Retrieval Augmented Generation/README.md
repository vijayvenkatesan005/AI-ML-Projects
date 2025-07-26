# Video Metadata RAG (Retrieval-Augmented Generation) App

This Streamlit application lets you interactively analyze and query metadata from YouTube videos using state-of-the-art language models. Just provide video URLs, and the app will:

- Extract rich metadata from each video (title, uploader, views, likes, duration, description, tags, etc.)
- Index these details for efficient semantic search using a Retrieval-Augmented Generation (RAG) pipeline
- Allow you to pose natural language questions about your uploaded videos, with answers grounded in their metadata—powered by OpenAI LLMs

---

## Features

- Paste one or more YouTube video URLs in the sidebar
- Metadata is automatically extracted and indexed
- Ask questions about any aspect of your uploaded videos (e.g. “Which video has the most views?”, “Tell me the uploaders”)
- Source attribution: see which video(s) any answer is based on

---

## Quickstart

1. **Clone the repository:**
    ```
    git clone https://github.com/vijayvenkatesan005/AI-ML-Projects.git
    cd "Retrieval Augmented Generation"
    ```

2. **Install the dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Add your OpenAI API key:**
    - Create a file named `secret_key.py` in the repo folder with this content:
      ```
      openai_key = "sk-..."   # Replace with your real OpenAI API key
      ```

4. **Run the Streamlit app:**
    ```
    streamlit run RAG.py
    ```

5. **Paste video URLs** (YouTube) into the sidebar. Wait for indexing, then ask questions in the main input box.

---

## Notes

- Works with standalone YouTube video URLs (not playlists).
- No OpenAI key included—get yours at [platform.openai.com](https://platform.openai.com/).
- Your `secret_key.py` should not be uploaded to GitHub for security reasons.
- All FAISS vector indexes and extracted metadata stay local to your environment.

---

## Example Use Cases

- Get a summary of all video descriptions
- Find the video with the highest like/view count
- List topics (tags) covered across all your videos
- Ask for uploader names, upload dates, durations, and more

---

## License

This project is for educational/research purposes. See LICENSE file.

---

*Questions or suggestions? Open an issue or pull request!*
