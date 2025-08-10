import semantic_kernel as sk

from plugins.Movies.tmdb import TMDbService


async def main():
    # Initialize the kernel
    kernel = sk.Kernel()

    tmdb_service = kernel.add_plugin(TMDbService(), "TMDBService")

    # Test the function with the inputs.
    # uncomment the line to test the function
    print(
        await tmdb_service["get_movie_genre_id"](
            kernel, sk.functions.KernelArguments(genre_name="action")
        )
    )
    print(
        await tmdb_service["get_tv_show_genre_id"](
            kernel, sk.functions.KernelArguments(genre_name="action")
        )
    )
    print(
        await tmdb_service["get_top_movies_by_genre"](
            kernel, sk.functions.KernelArguments(genre_name="action")
        )
    )
    print(
        await tmdb_service["get_top_tv_shows_by_genre"](
            kernel, sk.functions.KernelArguments(genre_name="action")
        )
    )
    print(await tmdb_service["get_movie_genres"](kernel, sk.functions.KernelArguments()))
    print(await tmdb_service["get_tv_show_genres"](kernel, sk.functions.KernelArguments()))


# Run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
