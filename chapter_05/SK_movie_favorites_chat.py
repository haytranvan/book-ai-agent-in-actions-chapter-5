import asyncio
import os
from dotenv import load_dotenv

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.contents.chat_history import ChatHistory

# Load environment variables
load_dotenv()

from plugins.Movies.tmdb import TMDbService
from plugins.SimpleFavorites.simple_favorites import SimpleFavoriteService

selected_service = "OpenAI"
kernel = sk.Kernel()

service_id = None
if selected_service == "OpenAI":
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")
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

# Add TMDB service and Favorites service
tmdb_service = kernel.add_plugin(TMDbService(), "TMDBService")
favorite_service = kernel.add_plugin(SimpleFavoriteService(), "FavoriteService")

# Import FunctionChoiceBehavior for auto function calling
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

# Set up the execution settings
if selected_service == "OpenAI":
    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id=model_id,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.8,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
elif selected_service == "AzureOpenAI":
    execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id=deployment,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.8,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

from semantic_kernel.functions import KernelFunctionFromPrompt

# Create comprehensive movie recommendation and favorites chat function
prompt_template_config = sk.prompt_template.PromptTemplateConfig(
    template="""You are a comprehensive movie recommendation and favorites management assistant. You can:

**Movie Discovery Features:**
1. Find top movies by genre using TMDB database
2. Get movie genre information
3. Search for specific movies and get details
4. Provide movie recommendations

**Favorites Management Features:**
1. Add movies to personal favorites list (saved to CSV file)
2. View all favorite movies
3. Get favorites filtered by specific genre
4. Manage personal movie watchlist

**Available Commands:**
- "Find top [genre] movies" - Discover best movies by genre
- "Show me action movies" - Get top action movies
- "Add [movie title] to favorites" - Save movie to personal collection
- "Show all my favorites" - View complete favorites list
- "Show my [genre] favorites" - Filter favorites by genre
- "What are the best horror movies?" - Get top movies in any genre

**Interaction Guidelines:**
- When users ask for movie recommendations, use TMDB to find top movies
- Always offer to add recommended movies to their favorites
- Help users manage their personal movie collection
- Provide detailed movie information when available
- Keep track of user preferences through their favorites

**Example Interactions:**
- User: "Find top action movies" → Show TMDB action movies + offer to add to favorites
- User: "Add Inception to favorites" → Add to CSV file
- User: "Show my sci-fi favorites" → Filter and display sci-fi movies from favorites

{{$user_input}}""",
    name="Chat",
    template_format="semantic-kernel",
    execution_settings=execution_settings,
)

chat_function = KernelFunctionFromPrompt(
    prompt_template_config=prompt_template_config,
    function_name="Chat",
    description="Comprehensive movie recommendation and favorites management assistant with TMDB integration and CSV storage"
)

# Add the chat function to kernel
kernel.add_plugin(chat_function, "ChatBot")

history = ChatHistory()

history.add_system_message("You are a comprehensive movie recommendation and favorites management assistant with TMDB database access and CSV-based favorites storage.")
history.add_user_message("Hi there, who are you?")
history.add_assistant_message(
    """Hello! I'm your comprehensive movie recommendation and favorites assistant! 🎬

**What I can do for you:**

🔍 **Movie Discovery**
• Find top movies by any genre using TMDB database
• Get detailed movie information and recommendations
• Search for specific movies

📝 **Favorites Management**
• Add movies to your personal favorites list
• View all your favorite movies
• Filter favorites by genre
• All data saved to CSV file for persistence

**Try these commands:**
• "Find top action movies"
• "Show me the best horror films"
• "Add The Matrix to my favorites"
• "Show all my favorites"
• "Show my comedy favorites"
• "What are the top sci-fi movies?"

What type of movies are you interested in today?"""
)

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
        "🎬 Movie Recommendation & Favorites Assistant 🌟\
\n  Type 'exit' to exit.\
\n  \
\n  🎯 CORE FEATURES:\
\n  • Discover top movies by genre (TMDB database)\
\n  • Personal favorites management with CSV storage\
\n  • Genre-based filtering and organization\
\n  \
\n  💡 EXAMPLE COMMANDS:\
\n  • 'Find top thriller movies'\
\n  • 'Show me the best action films'\
\n  • 'Add Avengers to my favorites'\
\n  • 'Show all my favorites'\
\n  • 'Show my romance favorites'\
\n  \
\n  🚀 Get started by asking for movie recommendations!"
    )
    while chatting:
        chatting = await chat()

if __name__ == "__main__":
    asyncio.run(main())
