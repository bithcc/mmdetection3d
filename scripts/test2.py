import os
def get_bin_files(directory):
    bin_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.bin'):
                bin_files.append(os.path.abspath(os.path.join(root, file)))
    return bin_files

bin_files = get_bin_files('/mnt/datasets/huichenchen/SemanticKitti/dataset/sequences')
print(len(bin_files))
   