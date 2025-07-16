with h5py.File('your_file.h5', 'r') as f:
    data = f['/group_name/dataset_name'][:]  # Replace with actual group/dataset name
    print(data)