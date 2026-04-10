import pkgutil
import importlib


IGNORE_LIST = ["utils", "__pycache__"]

for loader, name, is_pkg in pkgutil.walk_packages(__path__):

    if name in IGNORE_LIST:
        continue
    if is_pkg:
        full_data_name = f"{__name__}.{name}.{name.capitalize()}"
    else:
        full_data_name = f"{__name__}.{name.capitalize()}"

    # Import it! This triggers the @register
    try:
        importlib.import_module(full_data_name)
    except ModuleNotFoundError:
        # This happens if a folder exists but the matching .py file doesn't
        print(f"⚠️ Could not find the expected module {full_data_name}.py")
