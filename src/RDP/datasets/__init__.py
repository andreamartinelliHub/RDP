import pkgutil
import importlib

for loader, data_name, is_pkg in pkgutil.walk_packages(__path__):

    full_data_name = f"{__name__}.{data_name}"    
    # Import it! This triggers the @register
    importlib.import_module(full_data_name)