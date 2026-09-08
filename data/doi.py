"""DOI parsing constants shared by the acquisition and preparation layers.

This exists so that `prepare_data` can parse DOIs without importing `fetch_metadata`.
`fetch_metadata` reads API credentials from `_api_secrets` at import time, so anything that
imports it needs those credentials present, even to do work that never touches the network.
`DOI_PATTERN` is a plain regex with no such dependency, and keeping it here lets the
analysis path stay credential-free.
"""

# Matches a bare DOI (the "10.xxxx/yyyy" part), with or without a resolver prefix.
DOI_PATTERN = r'(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)'
