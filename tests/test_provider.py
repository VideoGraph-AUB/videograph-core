from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from videograph.utils import (
    get_api_provider,
    get_openai_client,
    resolve_model_name,
)


class ProviderConfigurationTests(unittest.TestCase):
    def test_openai_is_the_default_provider(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai"}, clear=True):
            self.assertEqual(get_api_provider(), "openai")
            self.assertEqual(resolve_model_name("openai/gpt-4o"), "gpt-4o")

    def test_openrouter_is_auto_selected_and_models_are_qualified(self) -> None:
        environment = {
            "OPENROUTER_API_KEY": "test-openrouter",
            "OPENROUTER_APP_NAME": "VideoGraph Tests",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(get_api_provider(), "openrouter")
            self.assertEqual(resolve_model_name("gpt-4o"), "openai/gpt-4o")
            self.assertEqual(
                resolve_model_name("text-embedding-3-small", "embedding"),
                "openai/text-embedding-3-small",
            )
            self.assertEqual(
                resolve_model_name("whisper-1", "transcription"),
                "openai/whisper-1",
            )

    def test_openrouter_client_uses_compatible_base_url(self) -> None:
        environment = {
            "VIDEOGRAPH_API_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-openrouter",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "videograph.utils.OpenAI"
        ) as client_class:
            get_openai_client()
        client_class.assert_called_once_with(
            api_key="test-openrouter",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"X-OpenRouter-Title": "VideoGraph"},
        )

    def test_explicit_openai_remains_available_when_both_keys_exist(self) -> None:
        environment = {
            "VIDEOGRAPH_API_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-openai",
            "OPENROUTER_API_KEY": "test-openrouter",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "videograph.utils.OpenAI"
        ) as client_class:
            get_openai_client()
            model = resolve_model_name("openai/gpt-4o", "chat")
        client_class.assert_called_once_with(api_key="test-openai")
        self.assertEqual(model, "gpt-4o")

    def test_invalid_provider_is_rejected(self) -> None:
        with patch.dict(
            os.environ, {"VIDEOGRAPH_API_PROVIDER": "invalid"}, clear=True
        ):
            with self.assertRaises(RuntimeError):
                get_api_provider()


if __name__ == "__main__":
    unittest.main()
