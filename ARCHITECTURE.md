# Lio Architecture v1

## مبادئ أساسية
1. Mobile app never stores OpenAI API keys.
2. Backend owns credentials, tools, orchestration, memory and approvals.
3. Lio speaks Arabic, German and English, with language expansion later.
4. Lio delegates to specialist agents for multi-step work.
5. Irreversible actions require explicit approval.
6. Device replacement must not lose Lio's memory; production data moves to cloud storage.

## Services
- Mobile Client
- Lio API Gateway
- Agent Orchestrator
- Research Agent
- Business Agent
- Communication Agent
- Monitoring Agent
- Memory Service
- Approval Service
- Future: Notification Service
- Future: Voice Service
- Future: Website Watch Worker
- Future: Gmail / Calendar / Files connectors

## Production roadmap
### Foundation
- Auth
- Cloud DB
- encrypted secrets
- audit log

### Intelligence
- Agents SDK
- Web Search
- file understanding
- task planner

### Voice
- push-to-talk
- STT
- TTS
- interruption / streaming

### Automation
- scheduled tasks
- website watchlist
- push notifications
- approval center

### Integrations
- Gmail
- Calendar
- cloud files
- business systems
