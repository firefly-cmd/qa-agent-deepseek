# Getting Started

## 1. Prerequisites

Before running the project, ensure you have the following installed:

Python 3.13+
Ollama: Follow the installation instructions for your operating system from the Ollama GitHub repository.

## 2. Set Up Ollama

### Download and Install Ollama:
Visit the Ollama GitHub repository and follow the installation instructions for your OS.

### Pull Required Models:
Open a terminal and run the following commands to download the required models:

'''
ollama pull deepseek-r1
ollama pull nomic-embed-text
'''

These models will be used for reasoning (deepseek-r1) and embeddings (nomic-embed-text).

### Start Ollama Server:
Run the following command to start the Ollama server:
bash
'''
ollama serve
'''

Keep this terminal window open while using the app.

## 3. Set Up the Project

### Install Python Dependencies:
Create a virtual environment (optional but recommended):
'''
python -m venv venv
source venv/bin/activate
'''

### Install the required Python packages:

'''
pip install -r requirements.txt
'''

## 4. Run the Streamlit App

## Start the App:
Run the following command to start the Streamlit app:

'''
streamlit run app.py
'''

This will open the app in your default web browser.

Upload Documents:
Use the sidebar to upload PDF documents. Click Ingest Documents to process and store them in ChromaDB.

Ask Questions:
Once documents are ingested, you can ask questions in the chat input box. The app will display the model’s reasoning process, answer, and source documents.

## Project Structure

Here’s an overview of the project files:

ingest.py: Handles document loading, splitting, and ingestion into ChromaDB.
retrieve.py: Retrieves relevant document chunks based on user queries.
qa.py: Combines retrieval and reasoning to generate answers using Deepseek R1.
app.py: The Streamlit app that ties everything together and provides the user interface.
requirements.txt: Lists the Python dependencies for the project.