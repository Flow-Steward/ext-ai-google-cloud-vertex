# Google Cloud Vertex Gemini provider

Connect an account-scoped Google Cloud project and Vertex AI location to a Flow Steward installation supporting extension runtime contract 1.4.0 or newer; the installer checks compatibility. This integration has not been verified with a live Google Cloud account.

## Connect

1. Select a Google Cloud project with billing enabled and [enable the Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com).
2. In [Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts), create a dedicated service account and grant it Vertex AI User (`roles/aiplatform.user`) on the project. Under its Keys tab, create a JSON key; your organization must allow service-account key creation.
3. In Flow Steward’s LLM provider panel, enter the Google Cloud project ID, a supported Vertex location such as `us-central1` or `global`, and the full service-account JSON key. Save the provider, discover models, verify their capabilities, and add the models you need. Gemini Developer API keys are not supported.

Google Cloud IAM, organization policy, model availability, quotas and charges apply. Discovery lists Google Model Garden publisher metadata, a global superset rather than an entitlement or location-specific inventory; a missing or empty response fails discovery to preserve the last catalog. Flow Steward’s capability verifier must approve each model before use. The provider uses the account's explicit credentials, preserves tool-call thought signatures during replay, and does not implement media generation or realtime interactions.

## References

- [Vertex generateContent API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.publishers.models/generateContent)
- [Regional and global endpoints](https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations)
- [Publisher model discovery](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1beta1/publishers.models/list)
- [Function calling and thought signatures](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tools/function-calling)
