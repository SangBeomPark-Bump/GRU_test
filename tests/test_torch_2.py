import time
def test_torch_2():
    aaa = time.time()
    import torch
    print(torch.__version__)
    bbb = time.time()
    print(bbb - aaa)