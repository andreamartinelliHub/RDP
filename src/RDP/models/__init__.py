import pkgutil
import importlib

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # print(loader, module_name, is_pkg)
    # Construct the full module path (e.g., 'models.chronos_v4')
    full_module_name = f"{__name__}.{module_name}"
    # print(full_module_name)
    
    # Import it! This triggers the @register
    importlib.import_module(full_module_name)