# from src.RDP.registry import Registry, MODEL_REGISTRY
# import src.RDP.models  # <--- This trigger the RDP

from RDP import models, Registry, MODEL_REGISTRY


def main():
    print("Hello from rdp!")
    print("\n Model Available in this Package:")
    print(MODEL_REGISTRY)

    print("\n Repeatable for datasets and configs!")
    Registry.show_all()

if __name__ == "__main__":
    main()
