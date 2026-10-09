Scenario6-eCommerceChatBot

Create a chatbot on a sample eCommerce Site
Step 1: Create a simple 4 product eCommerce Site - apples, oranges, grapes, strawberries
Step 2: Requirements Gathering on the project - mainly the products, the scope, guardrails (questions that can be asked or not asked)
Step 3: Select the model to be used - gpt-5.4-nano
Step 4: Create Implementation guidelines - use python, FastAPI
Step 5: Setup RAG and guidelines - take details from the src files and set up RAG using ChromaDB
Step 6: Add a widget on the website, so users can ask questions. This is the place where questions get picked up and answered by Agent
Step 7: Test the project

## Running

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Build the knowledge base (chunks Source/src1.txt and Source/src2.txt, embeds, stores in ChromaDB): `uv run ingest`
3. Start the site and API: `uv run ecommerce-chatbot`, then open site

