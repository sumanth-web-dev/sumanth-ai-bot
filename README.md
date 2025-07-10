# Multi-Tool AI Chatbot Backend

This is a Flask-based backend for an AI Chatbot that leverages various external tools to provide comprehensive answers and information. It uses LangChain for orchestrating tool calls and Groq for the underlying language model.

## Features

* **Intelligent Responses:** Utilizes a powerful language model to understand and respond to user queries.
* **Research Capabilities (Arxiv & Wikipedia):** Can search academic papers via Arxiv and general knowledge from Wikipedia.
* **Web Search (Tavily):** Performs real-time web searches to answer questions requiring up-to-date information.
* **Stock Price Lookup (Yahoo Finance):** Fetches the latest stock closing prices for given symbols.
* **Conversation Memory:** Maintains context throughout the chat session.

## Technologies Used

* **Flask:** Web framework for the backend API.
* **Flask-CORS:** Enables Cross-Origin Resource Sharing.
* **LangChain:** Framework for developing applications powered by language models.
* **LangGraph:** Used for building stateful, multi-step agent applications.
* **Groq:** Provides fast inference for the `qwen-qwq-32b` large language model.
* **Arxiv API Wrapper:** Integrates with the Arxiv academic paper database.
* **Wikipedia API Wrapper:** Integrates with the Wikipedia encyclopedia.
* **Tavily Search API:** Provides robust web search capabilities.
* **yfinance:** Python library for fetching historical market data from Yahoo Finance.
* **python-dotenv:** For managing environment variables.

## Setup Instructions

Follow these steps to set up and run the chatbot backend locally.

### Prerequisites

* Python 3.8+
* An active **Groq API Key**
* An active **Tavily API Key**

### 1. Clone the Repository (if applicable)

If your `app.py` is part of a larger project, ensure you have the project cloned.

### 2. Install Dependencies

Navigate to your project directory in the terminal and install the required Python packages:

```bash
pip install Flask Flask-Cors langchain-community langchain-groq langgraph yfinance python-dotenv
