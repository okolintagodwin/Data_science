## Birte — Conversational Forecasting Agent (MCP)

Birte is a conversational AI assistant that automates time-series forecasting through intelligent data preprocessing, real-world event discovery, and an MCP-based tool architecture. Users can upload messy data, request forecasts in natural language, and receive cleaned data, futureEXPERT forecasts, and relevant external events for context.

## Core Capabilities

**WP1 — Agent-Based Data Preprocessing**
An LLM-powered agent automatically cleans and structures user data by parsing dates, standardizing headers, handling missing values, restructuring columns, deduplicating rows, and resampling time series. It also generates a futureexpert_checkin_config.json file required for forecasting. The agent only asks the user for clarification when necessary.

**WP2 — Intelligent Event Discovery**
Birte enriches forecasts with real-world context by interpreting user intent via Gemini, generating search queries, retrieving articles from NewsAPI, filtering and clustering them into structured events, and optionally storing them in MongoDB.

**WP3 — MCP Server Integration**
An MCP server exposes Birte’s functionality as standardized tools:

preprocess_files: cleans raw time-series data

forecast_futureexpert: runs forecasts via futureEXPERT

search_events: discovers relevant external events

ask_birte: conversational router that selects the right tool

**Architecture**
The system is built around birte_mcp_server.py, which connects user requests to three main components: birte_preprocessing.py for data cleaning, forcasting.py for forecasting with futureEXPERT, and event.py for news intelligence and event extraction.

**Environment Variables**
The system requires the following environment variables: PROGAI_TOKEN, FUTUREEXPERT_USER, FUTUREEXPERT_PASSWORD, NEWS_API_KEY, GEMINI_API_KEY, and optionally MONGO_URI.

**Context**
This project was developed in collaboration with prognostica GmbH.
