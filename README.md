# Google Cloud Vertex Gemini provider (in development)

This extension is intended to connect an account-scoped Google Cloud project and Vertex AI location to Flow Steward. It depends on the proposed Core 1.4.0 LLM provider contract for separate plaintext `project_id` and `location` settings. It is not ready to install or use.

## Intended connection

Enable the Vertex AI API and billing for the project, grant a dedicated service account the permissions needed to use the selected models, and provide its service-account JSON as the account's encrypted provider credential. Choose a project ID and a supported Vertex location (including `global` where appropriate). Google Cloud IAM, organization policy, model availability, quotas, and charges apply. Do not use a Gemini Developer API key; that is a separate provider.

The runtime must construct refreshable credentials explicitly from that account's secret, never from application-default or ambient credentials. Model discovery lists Google Model Garden publisher metadata, a global superset rather than an entitlement or location-specific inventory. A missing or empty publisher response fails discovery to preserve the last catalog. Flow Steward’s existing capability verifier must approve each model before use. Tool-call thought signatures must be preserved with their function calls during replay. The provider does not implement media generation or realtime interactions.

## References

- [Vertex generateContent API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/projects.locations.publishers.models/generateContent)
- [Regional and global endpoints](https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations)
- [Publisher model discovery](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1beta1/publishers.models/list)
- [Function calling and thought signatures](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tools/function-calling)
