
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import Tool
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import yfinance as yf
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate



load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

app = Flask(__name__)
CORS(app)

# LangChain Tools Setup
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
tavily = TavilySearchResults(api_key=tavily_api_key, top_k_results=2)

def yahoo_finance_search(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")
    if not data.empty:
        price = data["Close"].iloc[-1]
        return f"Latest closing price for {symbol.upper()} is ₹{price:.2f}"
    else:
        return f"Could not fetch price for {symbol.upper()}. Please check the symbol."

yahoo_finance_tool = Tool.from_function(
    func=yahoo_finance_search,
    name="yahoo_finance_search",
    description="Fetches stock price for a given symbol from Yahoo Finance."
)

tools = [arxiv, wikipedia, tavily, yahoo_finance_tool]
llm = ChatGroq(model="qwen-qwq-32b", temperature=0.0, groq_api_key=groq_api_key)
llm_with_tools = llm.bind_tools(tools)
memory = ConversationBufferMemory(return_messages=True)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

prompt_template = PromptTemplate.from_template(
    """You are an intelligent AI assistant that helps users with their queries. 
You carefully reason step by step before responding.

Here is the conversation history so far:
{history}

Now, a new message has arrived from the user.

User: {user_message}

Please think through the problem step by step, explaining your reasoning clearly before providing a final answer.

Respond in the following format:

<thought>
[Your step-by-step reasoning here]
</thought>

<answer>
[Your final concise helpful answer here]
</answer>
"""
)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")


    memory.chat_memory.add_message(HumanMessage(content=user_message))

    history = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in memory.chat_memory.messages
    )


    formatted_prompt = prompt_template.format(history=history, user_message=user_message)


    response = graph.invoke({"messages": [HumanMessage(content=formatted_prompt)]})

    assistant_message = response["messages"][-1].content


    memory.chat_memory.add_message(AIMessage(content=assistant_message))

    return jsonify({"assistant_message": assistant_message}) 


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
