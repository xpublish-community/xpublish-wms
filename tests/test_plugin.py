import pytest

from xpublish.plugins import manage


def test_import_plugin():
    plugins = manage.load_default_plugins()
    print(plugins)
    assert 'cf_wms' in plugins