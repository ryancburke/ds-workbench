"""Keep one bad stored run from breaking a whole Workbench section."""
import functools
import logging

import dash_bootstrap_components as dbc

log = logging.getLogger(__name__)


def safe_render(fn):
    """Runs live in the browser's session storage, so a tab can hold results saved by an older
    version of a skill. If one can't be rendered, show a notice on that card instead of
    failing the section's callback."""
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            log.exception("%s could not render a stored run", fn.__name__)
            return dbc.Alert("This run was saved by an earlier version of the skill and can't be "
                             "displayed. Run it again to see the current output.",
                             color="secondary", className="py-2 small mb-0")
    return inner
