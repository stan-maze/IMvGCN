import os
import numpy as np


def load_arrays_from_folder(folder):
    npy_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    arrays = {}
    for file_name in npy_files:
        file_path = os.path.join(folder, file_name)
        array = np.load(file_path)
        arrays[file_name] = array
    return arrays


def print_array_info(arrays):
    for file_name, array in arrays.items():
        print("File:", file_name)
        print("Array:")
        print(array)
        print("Shape:", array.shape)
        print("Data Type:", array.dtype)
        print()


def main(args):
    # 目录路径
    folder_path = os.path.join(os.getcwd(), "data", "datasets", args.dataset)
    # 加载文件夹中的所有数组
    arrays = load_arrays_from_folder(folder_path)
    # 打印数组信息
    print_array_info(arrays)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="The path where the directory is located")
    parser.add_argument("--dataset", help="The name of the dataset directory")
    args = parser.parse_args()

    main(args)
