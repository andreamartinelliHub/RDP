import ast

def get_method_source(file_path, method_name):
    """Extracts the logic of a specific method from a file."""
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            # We use ast.unparse to turn the logic back into a standard string
            # This ignores comments and formatting differences!
            return ast.unparse(node)
    return None

def compare_model_logic(file1, file2):
    methods_to_check = ['__init__', 'forward']
    
    for method in methods_to_check:
        source1 = get_method_source(file1, method)
        source2 = get_method_source(file2, method)
        
        if source1 == source2:
            print(f"✅ {method} logic is identical.")
        else:
            print(f"❌ {method} logic has CHANGED.")