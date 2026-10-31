"""Offline production-function tests with fake search and provider clients."""

import ast
import asyncio
import importlib.util
import json
import os
import types
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "provider_clients", ROOT / "tobb_gpt/modules/provider_clients.py"
)
clients = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clients)


class RequestClient:
    def create(self, **kwargs):
        return kwargs

    async def acreate(self, **kwargs):
        return kwargs


class Provider:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.client = RequestClient()

    @classmethod
    def from_documents(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return cls()

    def as_retriever(self, **kwargs):
        return self


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "tobb_gpt/app.py"
        names = {
            "is_api_key_valid",
            "create_query_vector_store",
            "create_llm",
            "transform_question",
        }
        definitions = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.response = '{"query": "valid search"}'
        self.request = None

        def complete(**kwargs):
            self.request = kwargs
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=self.response)
                    )
                ]
            )

        self.ns = {
            "os": os,
            "uuid": uuid,
            "json": json,
            "st": types.SimpleNamespace(
                sidebar=types.SimpleNamespace(warning=lambda *a, **kw: None)
            ),
            "openai": types.SimpleNamespace(
                ChatCompletion=types.SimpleNamespace(create=complete)
            ),
            "Chroma": Provider,
            "OpenAIEmbeddings": Provider,
            "HuggingFaceHubEmbeddings": Provider,
            "ChatOpenAI": Provider,
            "HuggingFaceHub": Provider,
            "bind_openai_client": clients.bind_openai_client,
            "WebBaseLoader": lambda urls: types.SimpleNamespace(
                load=lambda: []
            ),
            "MarkdownTextSplitter": lambda **kwargs: types.SimpleNamespace(
                split_documents=lambda docs: ["text"]
            ),
        }
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]),
                str(source),
                "exec",
            ),
            self.ns,
        )
        Provider.calls = []

    def test_validation_does_not_mutate_environment(self):
        before = dict(os.environ)
        self.assertTrue(self.ns["is_api_key_valid"]("openai", "sk-a"))
        self.assertEqual(dict(os.environ), before)

    def test_query_rewrite_uses_explicit_key(self):
        self.assertEqual(
            self.ns["transform_question"]("openai", "question", "sk-a"),
            "valid search",
        )
        self.assertEqual(self.request["api_key"], "sk-a")

    def test_invalid_rewrite_falls_back_to_question(self):
        for response in [
            "not json",
            "[]",
            "null",
            '{"query": 42}',
            '{"query": " "}',
        ]:
            self.response = response
            with self.subTest(response=response):
                self.assertEqual(
                    self.ns["transform_question"](
                        "openai", "original", "sk-a"
                    ),
                    "original",
                )

    def test_empty_search_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "no usable"):
            self.ns["create_query_vector_store"](
                "openai", [{"Result": "No good result"}], "sk-a"
            )

    def test_query_collections_are_isolated_and_links_deduplicated(self):
        build = self.ns["create_query_vector_store"]
        results = [{"link": "https://example.com/a"}] * 2 + [
            {"snippet": "missing link"}
        ]
        _, urls = build("openai", results, "sk-a")
        build("openai", results, "sk-b")
        self.assertEqual(urls, ["https://example.com/a"])
        self.assertNotEqual(
            Provider.calls[0][1]["collection_name"],
            Provider.calls[1][1]["collection_name"],
        )

    def test_sync_requests_override_ambient_credentials(self):
        first = self.ns["create_llm"]("openai/test", "sk-a")
        second = self.ns["create_llm"]("openai/test", "sk-b")
        self.assertEqual(
            first.client.create(api_key="wrong")["api_key"], "sk-a"
        )
        self.assertEqual(second.client.create()["api_key"], "sk-b")

    def test_async_requests_override_ambient_credentials(self):
        bound = clients.CredentialBoundClient(RequestClient(), "sk-a")
        self.assertEqual(
            asyncio.run(bound.acreate(api_key="wrong"))["api_key"], "sk-a"
        )


if __name__ == "__main__":
    unittest.main()
