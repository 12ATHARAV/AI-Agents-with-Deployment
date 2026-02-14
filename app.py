import streamlit as st

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# UI
st.set_page_config(page_title="AI Content Agent", layout="wide")
st.title("AI Research & Social Media Agent (Groq)")

groq_key = st.sidebar.text_input("Groq API Key", type="password")
topic = st.text_input("Enter topic")

if groq_key and topic and st.button("Run Agent"):

    # Web Search
    search = DuckDuckGoSearchRun()
    search_content = search.invoke(topic)

    # LLM (Groq)
    llm = ChatGroq(
        api_key=groq_key,
        model="qwen/qwen3-32b",
        temperature=0.7
    )

    # Summary
    summary_prompt = PromptTemplate.from_template("""
    Summarize into 4–5 bullet points:

    {content}
    """)

    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"content": search_content})

    # Social prompts
    linkedin_prompt = PromptTemplate.from_template("""
    Write professional LinkedIn post with hashtags:

    {summary}
    """)

    twitter_prompt = PromptTemplate.from_template("""
    Write Twitter post under 280 chars with emojis:

    {summary}
    """)

    facebook_prompt = PromptTemplate.from_template("""
    Write friendly Facebook post:

    {summary}
    """)

    linkedin_chain = linkedin_prompt | llm | StrOutputParser()
    twitter_chain = twitter_prompt | llm | StrOutputParser()
    facebook_chain = facebook_prompt | llm | StrOutputParser()

    parallel = RunnableParallel({
        "LinkedIn": linkedin_chain,
        "Twitter": twitter_chain,
        "Facebook": facebook_chain
    })

    drafts = parallel.invoke({"summary": summary})

    st.subheader("Summary")
    st.write(summary)

    st.subheader("LinkedIn")
    st.write(drafts["LinkedIn"])

    st.subheader("Twitter")
    st.write(drafts["Twitter"])

    st.subheader("Facebook")
    st.write(drafts["Facebook"])
