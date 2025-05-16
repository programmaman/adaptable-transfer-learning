import importlib

def run(module_name):
    print(f"→ importing {module_name}...")
    mod = importlib.import_module(module_name)
    if hasattr(mod, "main"):
        print(f"✓ running {module_name}.main()")
        mod.main()
    elif hasattr(mod, "run"):
        print(f"✓ running {module_name}.run()")
        mod.run()
    elif hasattr(mod, "__dict__") and "__name__" in mod.__dict__:
        print(f"• {module_name} has no main/run; assuming import side-effects")
    else:
        raise RuntimeError(f"No entry point found in {module_name}")

if __name__ == "__main__":
    run("experiments.experiment_runner")
    run("experiments.struct_g_sweep")
    run("experiments.struct_g_analysis")
    print("✓ all stages finished")
