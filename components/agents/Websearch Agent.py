# components/generators/openai_search_preview_generator.py
import os, logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from haystack.core.component import component
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

@component
class OpenAISearchPreviewGenerator:
    def __init__(
        self,
        model: str = "gpt-4o-mini-search-preview-2025-03-11",
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: Optional[int] = None,
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv(api_key_env))
        self.max_output_tokens = max_output_tokens

    def run(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **_: Any
    ):
        oai_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.role.value
            if role == "tool":
                continue
            oai_msgs.append({"role": role, "content": m.text or ""})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_msgs,
        }
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        return {"replies": [ChatMessage.from_assistant(text)]}

logging.basicConfig(level=logging.WARNING)

load_dotenv()

system_prompt = (
    "You are a precise web search assistant. "
    "Search the web for the most up-to-date and relevant answers. "
    "Include URLs inline when possible."
)

agent = Agent(
    tools=[],
    chat_generator=OpenAISearchPreviewGenerator(
        model="gpt-4o-mini-search-preview-2025-03-11",
    ),
    system_prompt=system_prompt,
    max_agent_steps=1
)

user_q = "what's the date of today in persian calendar?"
result = agent.run(messages=[ChatMessage.from_user(user_q)])
print(result["messages"][-1].text)