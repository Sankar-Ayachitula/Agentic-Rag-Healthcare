# Frontend — Android

The client for MedAssist-RAG is a native **Android** app (Kotlin + Jetpack
Compose), not a web app. It talks to the FastAPI backend over HTTP.

Built after the backend is complete (Day 6+).

Planned:
- Kotlin + Jetpack Compose chat UI
- Retrofit (or Ktor) client calling the backend `POST /chat` endpoint
- Dev networking: emulator reaches host localhost via `10.0.2.2`;
  a physical device needs the machine's LAN IP or a deployed backend
- For demos: backend deployed to a public HTTPS URL

The Android project will live in its own module/repo (standard Android Studio
layout), so this folder is just a placeholder/notes for the client plan.
