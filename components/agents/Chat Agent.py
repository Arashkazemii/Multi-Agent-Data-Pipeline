import os
from dotenv import load_dotenv
from haystack.utils import Secret
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL")

# System prompt for the chat agent
chat_system_prompt = "You are a helpful assistant that can answer questions and provide information based on the context provided"

# Initialize the chat generator
chat_generator = OpenAIChatGenerator(
    api_key=Secret.from_token(openai_api_key), 
    model=openai_chat_model
    )

# Create the chat agent with the chat generator and system prompt
chat_agent = Agent(
    chat_generator=chat_generator,
    system_prompt=chat_system_prompt,
    tools=[],
    max_agent_steps=1
)

# Example usage of the chat agent
user_message = ChatMessage.from_user(text=input("User: "))
result = chat_agent.run(messages=[user_message])

print("Agent: "+result["messages"][-1].text)