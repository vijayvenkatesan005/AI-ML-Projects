AI Resume Screening & Candidate Fit Summarizer
This project is an interactive Streamlit web application to automate candidate shortlisting. It uses LLMs (OpenAI via LangChain) and advanced embeddings (Sentence Transformers) to rank uploaded resumes against a job description and generate concise, tailored explanations of candidate fit.

🚀 Features
Automated Resume Screening: Upload PDF resumes and instantly rank them for fit against any job description.

AI-Powered Fit Explanation: Each candidate receives a one-sentence, directly relevant explanation of their suitability crafted by an LLM using prompt engineering.

User-Centric Design: Upload multiple resumes at once, specify display count, and receive clear results in a table or downloadable CSV.

Robust Cloud Deployment: Runs on Streamlit Community Cloud—no local setup required.

🛠️ How It Works
Upload Candidate Resumes

Only PDF files named as firstName_lastName_resume.pdf are accepted for clarity and processing.

Paste Job Description

The job description text box is required to activate screening.

Candidate Ranking

The app splits and encodes both the job description and resumes using Sentence Transformers, then calculates cosine similarity scores.

LLM-Based Explanation

For each candidate, the app prompts an LLM to generate a one-sentence, direct summary, just the key skills/experience that match the role.

📦 Demo & Public Link
https://ai-candidate-screener.streamlit.app/

📋 Usage Instructions
Open the public app link above.

Paste a job description in the text box.

Upload one or more PDF resumes (follow firstName_lastName_resume.pdf naming).

Set how many top candidates to display.

Click to view results: ranked candidates and one-sentence AI fit explanations.

Download the results as a .csv file for full detail.

💡 Design Choices
Brevity by Design: Each explanation is purposefully limited to a single sentence. Recruiters are busy; concise summaries allow them to quickly assess fit without information overload.

Prompt Engineering: Prompts are crafted to request only actionable reasons, skipping phrases like "This candidate is good because..." for maximum information density.

Truncation Handling: Short explanations keep UI tidy, and if users want full text, they can download the data for detailed review.

🧩 Tech Stack
Streamlit (interactive web app)

Sentence Transformers (all-mpnet-base-v2)

LangChain & langchain-community (for document handling, LLM integration)

OpenAI API (LLM explanations)

Pandas, Numpy (data processing)

🔑 Setup & Requirements
Clone this repo:

bash git clone [https://github.com/your-username/your-repo.git](https://github.com/vijayvenkatesan005/AI-ML-Projects.git](https://github.com/vijayvenkatesan005/AI-ML-Projects.git)
Install dependencies:

bash
pip install -r requirements.txt
Set your OpenAI API key securely:

Locally: create a .streamlit/secrets.toml with

text
OPENAI_API_KEY = "sk-xxxx..."
On Streamlit Cloud: use the Secrets UI in app settings.

Run locally:

bash
streamlit run Candidate-Recommendation-Engine/main.py
⚠️ Notes
PDF parsing depends on correct filename conventions and valid content.

Dependencies are managed in requirements.txt—you do not need to install built-in modules like tempfile.

For demo purposes, LLM explanations are intentionally brief and can be further engineered for production.

📄 Example Output
Candidate	Similarity Score	Explanation
John Smith	0.78	5 years Python experience, REST API dev, led ML projects for fintech.
Priya Banerjee	0.75	AWS, SQL, and enterprise app development matching job requirements.
...	...	...
🙋♂️ Author
Vijay Venkatesan

Feel free to copy, adjust details, or let me know if you want any section rephrased or expanded!


