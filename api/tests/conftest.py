"""Shared fixtures: every test starts with fresh rate-limit budgets (all
suites share one TestClient host, "testclient")."""
import pytest

from api import guards


@pytest.fixture(autouse=True)
def _fresh_guards():
    guards.GUARDS.configure()
    yield
    guards.GUARDS.configure()
