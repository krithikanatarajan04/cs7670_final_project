from sources.fetcher import LocalHTMLFetcher

# Initialize
fetcher = LocalHTMLFetcher("corpus_index.json")

# Pick a URL from your index
test_url = "https://slowtravellerbangkok.com/silom-lumpini-neighborhood-guide"

# Fetch
content = fetcher.fetch([test_url])

# Print result
print(f"--- CONTENT FOR {test_url} ---")
print(content[test_url])
