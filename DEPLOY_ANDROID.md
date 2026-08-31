# Lio v5 — Deployment & Android Build

## What changed
- Correct default OpenAI model: `gpt-5.6-sol`.
- Dockerfile for the backend.
- Docker Compose for server/local deployment.
- CORS configuration.
- Android permissions for microphone and internet.
- EAS build profiles:
  - `preview` -> APK for direct Android installation.
  - `production` -> AAB for Google Play.

## Important
This package does NOT contain an APK yet.
An APK must be compiled in an Android build environment or through a build service.
The OpenAI API key must stay only in the backend `.env`.

## Backend deployment sequence
1. Copy `backend/.env.example` to `backend/.env`.
2. Add a valid `OPENAI_API_KEY` only on the server.
3. Build/run with Docker.
4. Verify `GET /health`.
5. Put the public HTTPS backend URL in the mobile environment.

## Android sequence
1. Configure the public backend URL.
2. Install mobile dependencies.
3. Compile the preview profile as APK.
4. Install APK on Android.
5. Grant microphone permission.
6. Test text chat, then voice, then Watch Center.

## What still blocks a live AI test?
A valid OpenAI API account/key with API billing enabled.
The application structure can continue to be built without that key.
