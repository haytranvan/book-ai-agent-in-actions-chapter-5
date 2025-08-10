import asyncio
import os
from dotenv import load_dotenv

import semantic_kernel as sk

# Load environment variables
load_dotenv()

selected_service = "OpenAI"
kernel = sk.Kernel()

service_id = None
if selected_service == "OpenAI":
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")  # Optional
    service_id = "oai_chat_gpt"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-3.5-turbo-1106",
            api_key=api_key,
            org_id=org_id,
        ),
    )
elif selected_service == "AzureOpenAI":
    from semantic_kernel.connectors.ai.azure_open_ai import AzureOpenAIChatCompletion

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    service_id = "aoai_chat_completion"
    kernel.add_service(
        AzureOpenAIChatCompletion(
            service_id=service_id,
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=api_key,
        ),
    )


# This function is currently broken
async def run_prompt():
    result = await kernel.invoke_prompt(prompt="recommend a movie about comedy")
    print(result)


# Use asyncio.run to execute the async function
asyncio.run(run_prompt())
