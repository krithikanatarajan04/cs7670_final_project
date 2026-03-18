import json
from abc import ABC, abstractmethod
from pathlib import Path
from bs4 import BeautifulSoup

class SourceFetcher(ABC):
    """Abstract base class for fetching content from URLs."""
    @abstractmethod
    def fetch(self, urls: list[str]) -> dict[str, str]:
        """Returns a mapping of URL to cleaned text content."""
        pass

class LocalHTMLFetcher(SourceFetcher):
    def __init__(self, index_path: str = "corpus_index.json"):
        self.index_file = self._resolve_path(index_path)
        
        # Fix: base_dir should be project root, not the index file's directory.
        # Index file lives at corpus/indices/xxx.json — going up 2 levels
        # gives us the project root where corpus/pages/ is also anchored.
        self.base_dir = self.index_file.parents[2]

        with open(self.index_file, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        
        self.url_to_path = {item["url"]: item["file_path"] for item in corpus_data}

    def _resolve_path(self, path_str: str) -> Path:
        """Helper to find the index file regardless of where the script is run."""
        path = Path(path_str)
        if path.is_absolute() and path.exists():
            return path
        
        # Check relative to current working directory or script location
        search_locations = [
            Path.cwd() / path_str,
            Path(__file__).resolve().parents[1] / path_str
        ]
        for loc in search_locations:
            if loc.exists():
                return loc
        raise FileNotFoundError(f"Could not find index file: {path_str}")

    def _clean_html(self, html_content: str) -> str:
        """Extracts clean prose from HTML body, removing noise."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Target the body exclusively
        body = soup.find("body")
        if not body:
            return ""

        # Remove non-prose elements (scripts, styles, nav, etc.)
        for noise in body(["script", "style", "nav", "footer", "header"]):
            noise.decompose()

        # Extract text with a separator to prevent words from sticking together
        raw_text = body.get_text(separator="\n")

        # Cleaning: Strip whitespace, remove empty lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        
        return "\n".join(lines)

    def fetch(self, urls: list[str]) -> dict[str, str]:
        """Reads files from disk based on the URL index."""
        results = {}
        for url in urls:
            if url not in self.url_to_path:
                print(f"Warning: URL not in index: {url}")
                continue

            # Resolve the HTML file path relative to the index file
            file_path = self.base_dir / self.url_to_path[url]

            if not file_path.exists():
                raise FileNotFoundError(f"Corpus file missing: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            results[url] = self._clean_html(html_content)
        
        return results