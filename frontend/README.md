# Frontend — Android

The client for MedAssist-RAG is a native **Android** app (Kotlin + Jetpack
Compose), not a web app. It talks to the FastAPI backend over HTTP.

The app lives in its own repo:
[Agentic-Rag-Healthcare-Android](https://github.com/Sankar-Ayachitula/Agentic-Rag-Healthcare-Android).

What it does:
- Kotlin + Jetpack Compose chat UI
- OkHttp SSE client streaming from the backend's `POST /chat/stream` endpoint
- Dev networking: emulator reaches host localhost via `10.0.2.2`;
  a physical device needs the machine's LAN IP or a deployed backend
- For demos: backend deployed to a public HTTPS URL

This folder is just notes; the Android Studio project is in the repo above.
