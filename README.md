# Google Cloud Vertex Gemini for Flow Steward

<img src="icon.png" width="96" alt="Google Cloud Vertex Gemini for Flow Steward">

[![CI](https://img.shields.io/github/actions/workflow/status/Flow-Steward/ext-ai-google-cloud-vertex/ci.yml?branch=main&label=CI)](https://github.com/Flow-Steward/ext-ai-google-cloud-vertex/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/Flow-Steward/ext-ai-google-cloud-vertex?label=release)](https://github.com/Flow-Steward/ext-ai-google-cloud-vertex/releases/latest)

Run your Flow Steward AI agents on **Google Gemini** through your own **Google Cloud project**. Chat, tool calling and structured JSON answers come from Vertex AI, billed to your Google Cloud account, with your data handled under your Google Cloud terms.

## What is this

[Vertex AI](https://cloud.google.com/vertex-ai) is Google Cloud's platform for Gemini models. [Flow Steward](https://docs.flowsteward.org) runs workflows and AI agents that connect your business systems.

This extension adds Google Cloud Vertex Gemini as a language-model provider in Flow Steward. Once it is connected, your agents can use Gemini models for conversations, for calling workflow tools and for answers that follow a JSON schema.

## What you can do

- **Run agents on Gemini.** Pick a Gemini model for any agent: support assistants, document readers, classifiers, routing agents.
- **Call workflow tools.** Gemini decides which tool to call and with which arguments; Flow Steward runs the tool and gives the result back to the model, keeping Gemini's reasoning context between steps.
- **Get structured answers.** Ask for a JSON object that matches your schema — an extracted invoice, a lead record, a decision with a reason — and get it validated before the workflow uses it.
- **See only the models you can use.** Discover lists the Gemini chat models your project can actually call in the location you chose, so every model you add works.
- **Keep costs under your control.** Everything is billed by Google to your project. Discover uses only Google's free token count; no text is generated until an agent runs.

## Why teams use it

- **Your cloud, your contract.** Requests go to Vertex AI in your Google Cloud project with your billing, quotas, data location and IAM.
- **Service-account access.** One JSON key with the Agent Platform User role — no personal Google account in the loop.
- **Global or regional.** Use the `global` location for the newest Gemini models, or a region when data must stay in one place.
- **Checked before use.** Flow Steward verifies chat, tool calling and structured output on each model before agents can use it.

## Get started

1. Create or choose a Google Cloud project and link a billing account to it.
2. Enable the [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com).
3. In [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts) create a service account with the role **Agent Platform User** (`roles/aiplatform.user`) and create a **JSON** key for it.
4. In Flow Steward open **Extensions → Google Cloud Vertex Gemini**, enter the Project ID, the location (`global` recommended) and the whole JSON key, and save.
5. Select **Discover**, then **Add** the models you want. The extension's **Google Cloud Vertex Gemini** page has the full step-by-step guide.

## Requirements

- A Google Cloud project with a linked billing account. The newest Gemini models need a full paid billing account.
- The Vertex AI API enabled in the project.
- Permission to create a service-account JSON key in the project (some organizations restrict it).
- Flow Steward with LLM provider extensions.

## FAQ

**Which location should I choose?**
`global`. The newest Gemini models are served there. The location is the Google data center that runs the model, not where your team works.

**Why does Discover show fewer models than Google's catalog?**
It lists only chat models your project can call in the chosen location. Speech, live audio, image and embedding models are not chat models, and models Google does not serve to your project are left out.

**Does Discover cost money?**
No. It reads Google's model catalog and checks each model with Google's free token count. Text is generated only when an agent runs.

**Can I use a Gemini API key from Google AI Studio?**
This extension connects through Google Cloud with a service account. A Google AI Studio key is a different product.

## In your language

**Русский.** Расширение подключает модели Google Gemini через ваш проект Google Cloud (Vertex AI) к агентам Flow Steward: диалоги, вызов инструментов workflow и ответы по JSON-схеме. Discover показывает только модели, доступные вашему проекту в выбранном регионе, и ничего не тратит.

**Lietuviškai.** Plėtinys prijungia Google Gemini modelius per jūsų Google Cloud projektą (Vertex AI) prie Flow Steward agentų: pokalbiai, workflow įrankių kvietimas ir atsakymai pagal JSON schemą.

**Polski.** Rozszerzenie podłącza modele Google Gemini przez Twój projekt Google Cloud (Vertex AI) do agentów Flow Steward: rozmowy, wywoływanie narzędzi workflow i odpowiedzi zgodne ze schematem JSON.

## Links

- [Vertex AI generateContent](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.publishers.models/generateContent)
- [Locations and the global endpoint](https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations)
- [Function calling](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tools/function-calling)
- [Flow Steward documentation](https://docs.flowsteward.org)
- [Report a problem](https://github.com/Flow-Steward/ext-ai-google-cloud-vertex/issues)

## For developers

The runtime (`runtime.py`) talks to Vertex AI over HTTPS with the service account's short-lived token. Google's error answers map to Flow Steward's provider categories in one place (`classify_error`); Discover filters the publisher catalog to chat models and checks each with `countTokens` in the configured location. Tests run with `python -m pytest tests`.

Google, Gemini, Google Cloud and Vertex AI are trademarks of Google LLC. This extension is not made or endorsed by Google.
