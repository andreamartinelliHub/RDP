import pkgutil
import importlib
# from pathlib import Path

# This automatically finds all .py files in the current folder
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Construct the full module path (e.g., 'models.chronos_v4')
    full_module_name = f"{__name__}.{module_name}"
    
    # Import it! This triggers the @register decorator inside the file
    importlib.import_module(full_module_name)