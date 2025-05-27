import scipy.io

ruta = "ecg_project/data/cpsc_dataset/A0001.mat"  # cambia a uno real
record = scipy.io.loadmat(ruta)
print("Claves del archivo:", record.keys())

for k in record:
    if k not in ['__header__', '__version__', '__globals__']:
        print(f"Contenido en {k}:")
        print(type(record[k]), record[k].shape)
