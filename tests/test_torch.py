
import time

def test_torch():
    aaa = time.time()
    import torch
    print(torch.__version__)
    bbb = time.time()
    print( bbb - aaa)