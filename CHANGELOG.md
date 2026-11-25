# Changelog

## 0.1.10

- Discover lists only the Gemini chat models your project can use in the configured location: speech, live audio, image, embedding and other non-chat models are left out, and each remaining model is checked with Google's free token count.
- Clear reasons when Google refuses: a project without billing or with a Free trial account shows as a billing problem, a model not served in the location as an unavailable model, and a request the model cannot handle as a capability mismatch.
- A step-by-step setup guide on the provider page: project, billing, API, service account, JSON key, fields, and what each message means.
- A new README.
