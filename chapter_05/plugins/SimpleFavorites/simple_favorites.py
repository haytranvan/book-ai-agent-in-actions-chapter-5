import csv
import os
import inspect
from datetime import datetime
from semantic_kernel.functions import kernel_function


def print_function_call():
    """Debug function to print function calls"""
    frame = inspect.currentframe()
    calling_frame = frame.f_back
    func_name = calling_frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(calling_frame)
    
    print(f"Function name: {func_name}")
    print("Arguments:")
    for arg in args:
        if arg != "self":
            print(f"  {arg} = {values[arg]}")


class SimpleFavoriteService:
    def __init__(self):
        self.csv_file = "favorite_movies_simple.csv"
        self.fieldnames = ['id', 'title', 'genre', 'added_date']
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def _load_favorites(self):
        """Load all favorite movies from CSV"""
        favorites = []
        try:
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    favorites.append(row)
        except FileNotFoundError:
            pass
        return favorites
    
    def _save_favorites(self, favorites):
        """Save all favorites back to CSV"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            for favorite in favorites:
                writer.writerow(favorite)
    
    def _get_next_id(self):
        """Get the next available ID"""
        favorites = self._load_favorites()
        if not favorites:
            return 1
        
        # Find the highest existing ID and add 1
        max_id = 0
        for fav in favorites:
            try:
                current_id = int(fav.get('id', 0))
                if current_id > max_id:
                    max_id = current_id
            except ValueError:
                continue
        
        return max_id + 1
    
    @kernel_function(
        description="Add a new movie to favorites list including genre",
        name="add_favorite_movie",
    )
    def add_favorite_movie(self, movie_title: str, genre: str = "") -> str:
        """
        Add a new movie to the favorites list.
        
        Parameters:
        - movie_title: Movie title (required)
        - genre: Movie genre (required)
        
        Returns:
        - Confirmation message
        """
        print_function_call()
        
        favorites = self._load_favorites()
        
        # Check if movie already exists
        for fav in favorites:
            if fav['title'].lower() == movie_title.lower():
                return f"Movie '{movie_title}' is already in favorites!"
        
        # Create new favorite entry
        new_favorite = {
            'id': str(self._get_next_id()),
            'title': movie_title,
            'genre': genre,
            'added_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        favorites.append(new_favorite)
        self._save_favorites(favorites)
        
        return f"Added '{movie_title}' to favorites! Total: {len(favorites)} movies"
    
    @kernel_function(
        description="Get all favorite movies",
        name="get_all_favorites",
    )
    def get_all_favorites(self) -> str:
        """
        Get all favorite movies.
        
        Returns:
        - List of all favorite movies
        """
        print_function_call()
        
        favorites = self._load_favorites()
        
        if not favorites:
            return "No movies in favorites list yet."
        
        result = f"FAVORITE MOVIES LIST ({len(favorites)} movies):\n\n"
        
        for fav in favorites:
            genre_info = f" | Genre: {fav['genre']}" if fav['genre'] else ""
            movie_id = fav.get('id', 'N/A')
            result += f"ID: {movie_id} - {fav['title']}{genre_info}\n"
            result += f"   Added: {fav['added_date']}\n\n"
        
        return result
    
    @kernel_function(
        description="Get favorite movies by genre",
        name="get_favorites_by_genre",
    )
    def get_favorites_by_genre(self, genre: str) -> str:
        """
        Get favorite movies filtered by genre.
        
        Parameters:
        - genre: Movie genre to filter by
        
        Returns:
        - List of movies in the specified genre
        """
        print_function_call()
        
        favorites = self._load_favorites()
        
        if not favorites:
            return "No movies in favorites list yet."
        
        # Filter by genre (case insensitive)
        genre_favorites = [f for f in favorites if genre.lower() in f['genre'].lower()]
        
        if not genre_favorites:
            return f"No '{genre}' movies found in favorites."
        
        result = f"FAVORITE {genre.upper()} MOVIES ({len(genre_favorites)} movies):\n\n"
        
        for fav in genre_favorites:
            movie_id = fav.get('id', 'N/A')
            result += f"ID: {movie_id} - {fav['title']}\n"
            result += f"   Added: {fav['added_date']}\n\n"
        
        return result
    
    @kernel_function(
        description="Delete a movie from favorites by ID or title",
        name="delete_favorite_movie",
    )
    def delete_favorite_movie(self, identifier: str) -> str:
        """
        Delete a movie from favorites list by ID or movie title.
        
        Parameters:
        - identifier: Movie ID (number) or movie title (string)
        
        Returns:
        - Confirmation message
        """
        print_function_call()
        
        favorites = self._load_favorites()
        
        if not favorites:
            return "No movies in favorites list to delete."
        
        # Try to find the movie by ID first, then by title
        movie_to_delete = None
        delete_index = -1
        
        # Check if identifier is a number (ID)
        try:
            movie_id = int(identifier)
            for i, fav in enumerate(favorites):
                if int(fav.get('id', 0)) == movie_id:
                    movie_to_delete = fav
                    delete_index = i
                    break
        except ValueError:
            # Not a number, search by title
            for i, fav in enumerate(favorites):
                if fav['title'].lower() == identifier.lower():
                    movie_to_delete = fav
                    delete_index = i
                    break
        
        if movie_to_delete is None:
            return f"Movie with identifier '{identifier}' not found in favorites."
        
        # Remove the movie
        favorites.pop(delete_index)
        self._save_favorites(favorites)
        
        movie_title = movie_to_delete['title']
        movie_id = movie_to_delete.get('id', 'N/A')
        
        return f"Deleted movie '{movie_title}' (ID: {movie_id}) from favorites! Remaining: {len(favorites)} movies"