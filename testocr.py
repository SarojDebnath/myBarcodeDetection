import cv2
import requests
from pyzbar import pyzbar

def serverDetect(url, imgPath, modelPath, conf):
    try:
        with open(modelPath, 'rb') as model_file, open(imgPath, 'rb') as image_file:
            files = {
                "model": model_file,
                "image": image_file
            }
            response = requests.post(url, files=files, timeout=10)
        status = response.status_code
        data = response.text  # Example response: '{"0":{"points":[801,442,823,527],"label":"plug","confidence":0.94}}'
        data = eval(data)
        det = []
        print(data)
        img = cv2.imread(imgPath)
        for key, value in data.items():
            if value['confidence'] >= conf:
                x = (value['points'][0] + value['points'][2]) / 2
                y = (value['points'][1] + value['points'][3]) / 2
                area = abs(value['points'][2] - value['points'][0]) * abs(value['points'][3] - value['points'][1])
                det.append([int(key), x, y, area])
                cv2.rectangle(img, (int(value['points'][0]), int(value['points'][1])),
                              (int(value['points'][2]), int(value['points'][3])), (0, 255, 0), 3)
                cv2.putText(img, value['label'], (int(value['points'][0]), int(value['points'][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(det)
        imgc=img.copy()
        #imgc=cv2.resize(imgc,(640,480))
        cv2.imshow('station', imgc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return status, det
    except Exception as e:
        print(e)
        return 500, []

def decode_barcodes(img, rois):
    barcodes = []
    for roi in rois:
        x, y, w, h = roi
        roi_img = img[y-2:y+h+2, x-2:x+w+2]
        decoded_objects = pyzbar.decode(roi_img)
        for obj in decoded_objects:
            barcodes.append(obj.data.decode('utf-8'))
    return barcodes

if __name__ == "__main__":
    url = "http://mgn-w-sarojd:8000/predict/v12"
    imgPath = "C:/Users/sarojd/Vision_Arsenal/QR_BAR/myBarcode/311.jpg"
    modelPath = "C:/Users/sarojd/Vision_Arsenal/QR_BAR/myBarcode/best.pt"
    conf = 0.0

    status, detections = serverDetect(url, imgPath, modelPath, conf)
    if status == 200:
        img = cv2.imread(imgPath)
        rois = [(int(det[1] - det[3] / 2), int(det[2] - det[3] / 2), int(det[3]), int(det[3])) for det in detections]
        barcodes = decode_barcodes(img, rois)
        print("Decoded Barcodes:", barcodes)
    else:
        print("Failed to detect objects.")