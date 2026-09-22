"""Test-session configuration.

``Settings`` reads ``.env``, so without this every test that builds one without
explicit arguments -- there are many, asserting the code defaults -- would see
whatever the developer happens to have in their ``.env``. That is how the suite
came to assert a real corpus as the unconfigured default, and how importing the
application started a bootstrap that fetched the configured URLs over the
network while the suite was still collecting.

Clearing one variable is enough because it disables the file rather than naming
settings to clear: see ``DOC_ETL_API_ENV_FILE`` in ``config.py``. A hand-kept
list of names would have to grow with every setting the application gains, and
would silently stop covering a test the moment it fell behind.

This is done at import rather than in a fixture on purpose: the ``Settings`` in
``doc_etl_api.config`` is built while the test module is being imported, so a
fixture -- which runs after collection -- would leave that object holding the
``.env`` values.

A test that wants a setting still sets it: an environment variable outranks the
file, and an explicit argument outranks both.
"""

import os

# Read before any test module imports the application.
os.environ["DOC_ETL_API_ENV_FILE"] = ""
