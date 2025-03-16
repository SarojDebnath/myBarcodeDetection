import depthai as dai
import cv2
import numpy as np
import subprocess

# Function to detect barcode ROIs using Sobel kernel
def detect_barcode_roi(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply horizontal Sobel filter to detect vertical edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.absolute(sobel_x)
    sobel_8u = np.uint8(abs_sobel_x / np.max(abs_sobel_x) * 255)
    
    # Threshold to binarize the image
    _, binary = cv2.threshold(sobel_8u, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up noise and connect barcode lines
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter contours: barcodes are typically narrow and tall
        if w > 50 and h > w * 2:  # Adjust these heuristics based on your barcode size
            rois.append((x, y, w, h))
    
    return rois

# Function to rectify barcode ROI for better decoding
def rectify_barcode(roi):
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find barcode boundaries
    edges = cv2.Canny(gray_roi, 100, 200)
    
    # Find contours in the ROI
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Get the largest contour (assumed to be the barcode)
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Define destination points for a flat rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    src_pts = box.astype("float32")
    
    # Compute perspective transform and warp
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(roi, M, (width, height))
    
    return rectified

# Function to decode barcode using zxing-cpp
def decode_barcode_zxing(image):
    try:
        # Save the rectified image temporarily
        cv2.imwrite("temp_barcode.png", image)
        
        # Call zxing-cpp (adjust path/command as per your setup)
        result = subprocess.run(["zxing-cpp", "temp_barcode.png"], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error decoding barcode: {str(e)}"

# Main function to set up DepthAI and process frames
def main():
    # Create DepthAI pipeline
    pipeline = dai.Pipeline()

    # Define color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)  # 4056x3040
    cam_rgb.setFps(30)  # Adjust FPS as needed
    cam_rgb.initialControl.setManualFocus(160)  # Set focus to 160

    # Output stream
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        print("Connected to OAK-1 Max. Streaming at 4056x3040...")
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        while True:
            # Get frame from the camera
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            # Step 2: Detect barcode ROIs
            rois = detect_barcode_roi(frame)
            
            # Step 3: Process each ROI, rectify, and decode
            for (x, y, w, h) in rois:
                roi = frame[y:y+h, x:x+w]
                rectified_roi = rectify_barcode(roi)
                if rectified_roi is not None:
                    barcode_data = decode_barcode_zxing(rectified_roi)
                    print(f"Decoded barcode at ({x}, {y}): {barcode_data}")
                    
                    # Optional: Draw rectangle around detected barcode
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, barcode_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)
            
            # Display the frame for debugging
            cv2.imshow("Barcode Detection", frame)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                break

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")