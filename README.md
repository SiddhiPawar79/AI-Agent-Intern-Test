# AI Agent Intern Test — RAG + Order Lookup Agent

# Overview
This project implements a customer-support AI agent using Retrieval-Augmented Generation (RAG) over the supplied Markdown knowledgebase and a secure order-status lookup tool using orders.json.
The system is designed to:
- Retrieve only relevant knowledgebase passages.
- Preserve document metadata and source information.
- Prefer active and authoritative policy documents.
- Provide source references for policy/product answers.
- Safely handle order-status questions.
- Maintain relevant conversation context across multiple turns.
- Avoid exposing internal customer/order information.
- Abstain or recommend human assistance when information is insufficient or conflicting.
- Provide an evaluation suite covering retrieval, groundedness, tool use, privacy, and multi-turn behavior.
- Provide structured debug logging for development and evaluation.

# 1. Features

# Retrieval-Augmented Generation
- Reads the supplied Markdown files from `knowledge-base/`.
- Extracts document front matter.
- Preserves metadata such as:
  - `document_id`
  - `title`
  - `status`
  - `effective_date`
  - `last_reviewed`
  - `audience`
  - `policy_authority`
  - `supersedes`
- Splits documents into searchable chunks.
- Generates semantic embeddings.
- Uses FAISS for local vector retrieval.
- Returns relevant passages instead of the complete knowledgebase.
- Includes filename and heading as source references.
- Prioritizes active and official policies over legacy/internal content.

# Order Lookup
The agent uses orders.json through an order lookup function.
The lookup:
- Requires an order ID when order information is needed.
- Accepts harmless variations such as lowercase IDs and surrounding whitespace.
- Handles malformed and unknown IDs safely.
- Uses the current `status` field as authoritative.
- Does not invent unavailable delivery estimates.
- Does not expose:
  - customer email
  - customer address
  - internal notes
  - risk scores
  - other internal-only fields
- Does not claim a lookup occurred if no lookup was performed.

# Multi-turn Conversation
The agent maintains relevant session context for follow-up questions.
Examples:
User: Do you ship internationally?
Agent: ...
User: What about Canada?
Agent: ...
