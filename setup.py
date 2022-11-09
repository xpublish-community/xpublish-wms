from setuptools import setup

# CHANGE PKG NAME HERE
pkg_name = "ioos_pkg_skeleton"

setup(
    use_scm_version={
        "write_to": f"{pkg_name}/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
