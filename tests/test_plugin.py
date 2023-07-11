from xpublish.plugins import manage

import xpublish_wms


def test_import_plugin():
    plugins = manage.configure_plugins({"cf_wms": xpublish_wms.CfWmsPlugin})
    assert "cf_wms" in plugins
