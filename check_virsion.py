try:
    import torch
    import matplotlib
    import memory_profiler
    import numpy
    import pandas
    import gym

    print("torch version:", torch.__version__)
    print("matplotlib version:", matplotlib.__version__)
    print("memory_profiler version:", memory_profiler.__version__)
    print("numpy version:", numpy.__version__)
    print("pandas version:", pandas.__version__)
    print("gym version:", gym.__version__)

except ImportError as e:
    print(f"Module not found: {e.name}. Make sure it is installed.")
