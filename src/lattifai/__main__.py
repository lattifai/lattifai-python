import os
import warnings

if "NEMORUN_SHOW_WARNINGS" not in os.environ:
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

from nemo_run.cli.api import create_cli  # noqa: E402

app = create_cli()
