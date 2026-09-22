## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

```bash
semble search "authentication flow" ./my-project --max-snippet-lines 10  # first 10 lines only, concise
semble search "save_pretrained" ./my-project                          # full chunk content
semble search "save model to disk" ./my-project --top-k 10           # more results
```

The index is built on first run (and cached for subsequent runs) and invalidated automatically when files change.

Use `--content docs` to search documentation and prose, `--content config` for config files (yaml, toml, etc.), or `--content all` to search code, docs, and config:

```bash
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble search "authentication" ./my-project --content all
```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

```bash
semble find-related src/auth.py 42 ./my-project
```

`path` defaults to the current directory when omitted; git URLs are accepted. Pass several paths or URLs to search related repos together, e.g. when this project calls a service defined in a sibling repo. Result paths are then prefixed with the repo name and the output includes a `repos` map from prefix to location:

```bash
semble search "invoice endpoint" ./service-a ../service-b