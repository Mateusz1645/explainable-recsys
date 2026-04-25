from pathlib import Path


def resolve_project_root(cwd: Path | None = None) -> Path:
    """Resolve project root from current working directory.

    If a ``data`` directory exists in ``cwd``, that directory is treated as
    project root. Otherwise, the parent directory is returned.
    """
    current = cwd or Path.cwd()
    return current if (current / "data").exists() else current.parent


def get_file_map(project_root: Path | None = None) -> dict[str, Path]:
    """Build a mapping of logical table names to raw ``.dat`` file paths."""
    root = project_root or resolve_project_root()
    raw_dir = root / "data" / "raw"
    return {
        "movie_actors": raw_dir / "movie_actors.dat",
        "movie_countries": raw_dir / "movie_countries.dat",
        "movie_directors": raw_dir / "movie_directors.dat",
        "movie_genres": raw_dir / "movie_genres.dat",
        "movie_locations": raw_dir / "movie_locations.dat",
        "movie_tags": raw_dir / "movie_tags.dat",
        "movies": raw_dir / "movies.dat",
        "tags": raw_dir / "tags.dat",
        "user_ratedmovies": raw_dir / "user_ratedmovies.dat",
        "user_taggedmovies": raw_dir / "user_taggedmovies.dat",
    }
