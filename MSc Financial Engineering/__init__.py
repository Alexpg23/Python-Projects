from arch._version import get_versions
from arch.univariate.mean import arch_model
from arch.utility import test

__version__ = get_versions()["version"]
del get_versions


def doc() -> None:
    import webbrowser

    webbrowser.open("https://bashtage.github.io/arch/")


__all__ = ["arch_model", "__version__", "doc", "test"]
