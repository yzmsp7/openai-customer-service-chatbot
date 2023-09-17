# 📦 OpenAI Customer Service Chatbot

This project try to build up a customer serivce chatbot for taiwan mobile company.  
I use streamlit as my chat interface and openai api as backend.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://openai-customer-service-chatbot-twnmobile.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/yzmsp7/openai-customer-service-chatbot)

## Architecture

![system architecture](img/architecture.png)

- crawler_embedding.py
  - web crawler from [link](https://www.taiwanmobile.com/cs/public/faq/queryList.htm)
  - save as csv
  - convert to embedding using OpenAI API
- streamlit_app.py
  - main 
  - search similar question from database
  - answer the question


