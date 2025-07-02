import sys
import importlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def import_module_from_path(module_name: str, module_path: str) -> None:
    """
    Import a module from a given path and assign it a specified name.

    Args:
        module_name: Name to assign to the imported module.
        module_path: Path to the module being imported.

    Raises:
        ValueError: If the module has already been imported.
        FileNotFoundError: If the `__init__.py` file is not found in the module path.
    """
    # Based on https://stackoverflow.com/a/41595552.

    if module_name in sys.modules:
        logger.warning(f"{module_name} has already been imported as module.")
        return

    module_path = Path(module_path).resolve() / "__init__.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"No `__init__.py` in `{module_path}`.")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    logger.info(f"Imported {module_path.parent} as module '{module_name}'.")
