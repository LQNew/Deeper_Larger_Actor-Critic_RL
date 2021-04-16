import zlib
import pickle
import base64

import os
import sys
abs_path = os.path.abspath('.') # 表示当前所处的文件夹的绝对路径
sys.path.append(abs_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('encoded_thunk')
    args = parser.parse_args()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()