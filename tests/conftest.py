import pytest

from layout_redact import build_template
from layout_redact import synthetic as sy


@pytest.fixture(scope="session")
def reference():
    return sy.render_form(blank=True)


@pytest.fixture(scope="session")
def template(reference):
    return build_template(reference, sy.sample_regions(), "sample-order-form")
