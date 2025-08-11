import asyncio
import os
from dotenv import load_dotenv

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

# from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptTemplate
# These utilities are deprecated in SK 1.x
# from semantic_kernel.connectors.ai.open_ai.utils import (
#     chat_completion_with_tool_call,
#     get_tool_call_object,
# )
from semantic_kernel.contents.chat_history import ChatHistory

# Load environment variables
load_dotenv()

from plugins.Movies.tmdb import TMDbService

selected_service = "OpenAI"
kernel = sk.Kernel()

service_id = None
if selected_service == "OpenAI":
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")  # Optional
    service_id = "oai_chat_gpt"
    model_id = "gpt-4-1106-preview"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id=model_id,
            api_key=api_key,
            org_id=org_id,
        ),
    )
elif selected_service == "AzureOpenAI":
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    service_id = "aoai_chat_completion"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=api_key,
        ),
    )

tmdb_service = kernel.add_plugin(TMDbService(), "TMDBService")

# set up the execution settings
if selected_service == "OpenAI":
    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id=model_id,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.8,
    )
elif selected_service == "AzureOpenAI":
    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id=deployment,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.8,
    )

# OpenAIChatPromptTemplate
# prompt_template_config = sk.PromptTemplateConfig(
#     template="{{$user_input}}",
#     name="Chat",
#     template_format="semantic-kernel",
#     input_variables=[
#         InputVariable(
#             name="user_input", description="The user input", is_required=True
#         ),
#         InputVariable(
#             name="history",
#             description="The history of the conversation",
#             is_required=True,
#             default="",
#         ),
#     ],
#     execution_settings={"default": execution_settings},
# )

prompt_config = sk.prompt_template.PromptTemplateConfig(
    template="{{$user_input}}",
    name="Chat",
    template_format="semantic-kernel",
    execution_settings={"default": execution_settings},
)
from semantic_kernel.functions import KernelFunctionFromPrompt

chat_function = KernelFunctionFromPrompt(
    prompt="{{$user_input}}",
    function_name="Chat",
    description="Chat with the movie recommendation bot"
)


history = ChatHistory()

history.add_system_message("You recommend movies and TV Shows.")
history.add_user_message("Hi there, who are you?")
history.add_assistant_message(
    "I am Rudy, the recommender chat bot. I'm trying to figure out what people need."
)

# chat_function is already defined above


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    arguments = sk.functions.KernelArguments(
        user_input=user_input,
        history=("\n").join([f"{msg.role}: {msg.content}" for msg in history]),
    )
    result = await kernel.invoke(chat_function, arguments)
    print(f"GPT Agent:> {result}")
    return True


async def main() -> None:
    chatting = True
    print(
        "Welcome to the chat bot!\
\n  Type 'exit' to exit.\
\n  Try a math question to see the function calling in action (i.e. what is 3+3?)."
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
