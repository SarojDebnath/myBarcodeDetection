from pyzxing import BarCodeReader

reader = BarCodeReader()
results = reader.decode('310.jpg')
# Or file pattern for multiple files
#results = reader.decode('/PATH/TO/FILES/*.png')
print(results)
# Or a numpy array
# Requires additional installation of opencv
# pip install opencv-python
#results = reader.decode_array(img)