import cv2, zxingcpp

img = cv2.imread('1.JPG')
barcodes = zxingcpp.read_barcodes(img)
for barcode in barcodes:
	print('Found barcode:'
		f'\n Text:    "{barcode.text}"'
		f'\n Format:   {barcode.format}'
		f'\n Content:  {barcode.content_type}'
		f'\n Position: {barcode.position}')
if len(barcodes) == 0:
	print("Could not find any barcode.")