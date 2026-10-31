"""Bind request credentials for the repository's legacy OpenAI integration."""


class CredentialBoundClient:
    """Prevent requests from falling back to mutable openai.api_key state."""

    def __init__(self, client, api_key):
        self._client = client
        self._api_key = api_key

    def create(self, **kwargs):
        kwargs["api_key"] = self._api_key
        return self._client.create(**kwargs)

    async def acreate(self, **kwargs):
        kwargs["api_key"] = self._api_key
        return await self._client.acreate(**kwargs)


def bind_openai_client(provider, api_key):
    # LangChain 0.0.186 stores credentials in module globals during construction.
    # Its client field is replaceable; inject the key into every actual request.
    provider.client = CredentialBoundClient(provider.client, api_key)
    return provider
