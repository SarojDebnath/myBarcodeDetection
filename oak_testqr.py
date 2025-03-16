import threading
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import os
from pyzbar.pyzbar import decode
import zxingcpp  # Import zxing-cpp for better barcode decoding
sys.path.append("C:/Users/sarojd/Vision_Arsenal/QR_BAR/")
from OCR_Client import call_server as ocr
absolutely_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(absolutely_path)


class CameraStream:
    def __init__(self, previewSize, fps):
        self.size = previewSize
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRgb.setStreamName("rgb")
        self.camRgb.setPreviewSize(previewSize[0], previewSize[1])
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4000X3000)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.initialControl.setManualFocus(160)
        self.camRgb.setFps(fps)
        self.camRgb.preview.link(self.xoutRgb.input)
        
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        size = self.size
        with dai.Device(self.pipeline) as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            while not self.stopped:
                inRgb = qRgb.get()
                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                    frame = cv2.resize(frame, (size[0], size[1]))
                    self.frame = frame
                time.sleep(0.01)  # Small sleep to prevent high CPU usage

    def get_frame(self):
        return self.frame

    def stop(self):
        self.stopped = True

def barcode_reader(image):
    # Create a copy of the original image
    original_image = image.copy()
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using modern OpenCV syntax
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    
    # Subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # Blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    
    # Try different kernel sizes to better separate closely positioned barcodes
    kernel_sizes = [(21, 7), (15, 5), (25, 5)]
    all_contours = []
    
    for kernel_size in kernel_sizes:
        # Construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        
        # Find the contours in the thresholded image
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add contours to our collection
        all_contours.extend(contours)
    
    # Display the processed image for debugging
    cv2.imshow('Processed Image', closed)
    
    # If no contours were found, try direct decoding
    if not all_contours:
        return try_direct_decoding(image, original_image)
    
    # Initialize list to store decoded barcodes
    decoded_barcodes = []
    
    # Sort contours by area, largest first
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
    
    # Remove duplicate contours (those that overlap significantly)
    filtered_contours = []
    for c in all_contours:
        is_duplicate = False
        c_rect = cv2.boundingRect(c)
        
        for existing_c in filtered_contours:
            existing_rect = cv2.boundingRect(existing_c)
            
            # Calculate overlap
            x_overlap = max(0, min(c_rect[0] + c_rect[2], existing_rect[0] + existing_rect[2]) - max(c_rect[0], existing_rect[0]))
            y_overlap = max(0, min(c_rect[1] + c_rect[3], existing_rect[1] + existing_rect[3]) - max(c_rect[1], existing_rect[1]))
            overlap_area = x_overlap * y_overlap
            c_area = c_rect[2] * c_rect[3]
            
            # If overlap is more than 50% of the contour area, consider it a duplicate
            if overlap_area > 0.5 * c_area:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_contours.append(c)
    
    # Process all filtered contours
    for i, c in enumerate(filtered_contours):
        # Skip very small contours
        if cv2.contourArea(c) < 500:  # Reduced minimum area threshold
            continue
            
        # Compute the rotated bounding box of the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate aspect ratio of the rectangle
        width = rect[1][0]
        height = rect[1][1]
        
        # Make sure width is the longer side
        if width < height:
            width, height = height, width
            
        # Skip if dimensions are too small
        if width < 80 or height < 15:  # Reduced size thresholds
            continue
            
        # Calculate aspect ratio
        aspect_ratio = width / (height + 0.01)  # Avoid division by zero
        
        # Only process rectangles with appropriate aspect ratio for barcodes
        if 1.5 < aspect_ratio < 25.0:  # More permissive aspect ratio range
            # Draw the contour on the image
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            
            # Expand the ROI by 20% in each direction to ensure the full barcode is captured
            # Calculate the center of the rectangle
            center_x, center_y = rect[0]
            # Expand width and height
            expanded_width = width * 1.2
            expanded_height = height * 1.2
            # Create expanded rectangle
            expanded_rect = ((center_x, center_y), (expanded_width, expanded_height), rect[2])
            expanded_box = cv2.boxPoints(expanded_rect)
            expanded_box = np.int0(expanded_box)
            
            # Draw the expanded contour on the image with a different color
            cv2.drawContours(image, [expanded_box], -1, (255, 0, 0), 1)
            
            # Extract the expanded ROI
            # Get the transformation matrix
            src_pts = np.float32(expanded_box).reshape(4, 2)
            dst_pts = np.float32([[0, 0], [expanded_width, 0], [expanded_width, expanded_height], [0, expanded_height]])
            
            # Apply the perspective transformation
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            roi = cv2.warpPerspective(original_image, M, (int(expanded_width), int(expanded_height)))
            
            # Try to decode with zxingcpp
            try:
                # Use zxing-cpp without specifying formats, but with try_harder=True
                zxing_results = zxingcpp.read_barcodes(roi, try_harder=True)
                
                if zxing_results:
                    for zxing_result in zxing_results:
                        barcode_data = zxing_result.text
                        barcode_type = str(zxing_result.format)
                        
                        # Skip if not a 1D barcode format
                        if "QR" in barcode_type or "DATAMATRIX" in barcode_type or "AZTEC" in barcode_type:
                            continue
                        
                        # Check if the barcode matches one of our expected patterns
                        if (barcode_data == "0061003008" or 
                            barcode_data.startswith("FA7025") or 
                            barcode_data.startswith("LBADVA") or
                            len(barcode_data) > 0):  # Accept any non-empty barcode
                            
                            # Add to our tracking list
                            decoded_barcodes.append((barcode_type, barcode_data))
                            
                            print(f"Barcode found in ROI! Type: {barcode_type}, Data: {barcode_data}")
                            
                            # Add text to the main image
                            cv2.putText(image, f"{barcode_type}: {barcode_data}", 
                                      (box[0][0], box[0][1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error with zxing on ROI: {e}")
                
                # If zxing fails, try with pyzbar as a fallback
                try:
                    pyzbar_results = decode(roi)
                    if pyzbar_results:
                        for pyzbar_result in pyzbar_results:
                            barcode_data = pyzbar_result.data.decode('utf-8')
                            barcode_type = pyzbar_result.type
                            
                            # Add to our tracking list
                            decoded_barcodes.append((barcode_type, barcode_data))
                            
                            print(f"Barcode found with pyzbar in ROI! Type: {barcode_type}, Data: {barcode_data}")
                            
                            # Add text to the main image
                            cv2.putText(image, f"{barcode_type}: {barcode_data}", 
                                      (box[0][0], box[0][1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as pyzbar_error:
                    print(f"Error with pyzbar on ROI: {pyzbar_error}")
    
    # If no barcodes were found with the contour approach, try direct decoding
    if not decoded_barcodes:
        return try_direct_decoding(image, original_image)
    
    # Display a summary of all decoded barcodes at the top of the image
    if decoded_barcodes:
        # Count occurrences of each barcode
        barcode_counts = {}
        for barcode_type, barcode_data in decoded_barcodes:
            key = f"{barcode_type}:{barcode_data}"
            if key in barcode_counts:
                barcode_counts[key] += 1
            else:
                barcode_counts[key] = 1
        
        # Display summary at the top of the image
        y_offset = 30
        for i, (key, count) in enumerate(barcode_counts.items()):
            barcode_type, barcode_data = key.split(":", 1)
            summary_text = f"Found {count}x {barcode_type}: {barcode_data}"
            cv2.putText(image, summary_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
    
    return image

def try_direct_decoding(image, original_image):
    """Try direct decoding on the full image using zxing-cpp"""
    decoded_barcodes = []
    
    # Try with zxingcpp on the full image
    try:
        # Use zxing-cpp without specifying formats, but with try_harder=True
        zxing_results = zxingcpp.read_barcodes(original_image, try_harder=True)
        
        if zxing_results:
            for zxing_result in zxing_results:
                barcode_data = zxing_result.text
                barcode_type = str(zxing_result.format)
                
                # Skip if not a 1D barcode format
                if "QR" in barcode_type or "DATAMATRIX" in barcode_type or "AZTEC" in barcode_type:
                    continue
                
                # Check if the barcode matches one of our expected patterns or is non-empty
                if (barcode_data == "0061003008" or 
                    barcode_data.startswith("FA7025") or 
                    barcode_data.startswith("LBADVA") or
                    len(barcode_data) > 0):  # Accept any non-empty barcode
                    
                    # Add to our tracking list
                    decoded_barcodes.append((barcode_type, barcode_data))
                    
                    print(f"Barcode found on full image! Type: {barcode_type}, Data: {barcode_data}")
                    
                    # Try to get position information
                    try:
                        position = zxing_result.position
                        points = []
                        for i in range(4):
                            points.append((int(position[i].x), int(position[i].y)))
                        
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        # Draw the original bounding box
                        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                        
                        # Calculate expanded bounding box (20% larger)
                        # Find the center of the points
                        center_x = sum(p[0] for p in points) / 4
                        center_y = sum(p[1] for p in points) / 4
                        
                        # Expand points from center
                        expanded_points = []
                        for px, py in points:
                            # Vector from center to point
                            vx, vy = px - center_x, py - center_y
                            # Expand by 20%
                            expanded_points.append((int(center_x + vx * 1.2), int(center_y + vy * 1.2)))
                        
                        # Draw expanded bounding box
                        expanded_pts = np.array(expanded_points, np.int32)
                        expanded_pts = expanded_pts.reshape((-1, 1, 2))
                        cv2.polylines(image, [expanded_pts], True, (255, 0, 0), 1)
                        
                        # Add text near the barcode
                        cv2.putText(image, f"{barcode_type}: {barcode_data}", 
                                  (points[0][0], points[0][1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except:
                        # If position information is not available, just add text at the top
                        cv2.putText(image, f"{barcode_type}: {barcode_data}", 
                                  (10, 30 + 30 * len(decoded_barcodes)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error with zxing on full image: {e}")
    
    # Try different preprocessing techniques if no barcodes found
    if not decoded_barcodes:
        # Try with different preprocessing techniques
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Try with different threshold values
        for thresh_val in [100, 150, 200]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            try:
                zxing_results = zxingcpp.read_barcodes(binary, try_harder=True)
                
                if zxing_results:
                    for zxing_result in zxing_results:
                        barcode_data = zxing_result.text
                        barcode_type = str(zxing_result.format)
                        
                        # Skip if not a 1D barcode format
                        if "QR" in barcode_type or "DATAMATRIX" in barcode_type or "AZTEC" in barcode_type:
                            continue
                        
                        # Check if the barcode matches one of our expected patterns or is non-empty
                        if (barcode_data == "0061003008" or 
                            barcode_data.startswith("FA7025") or 
                            barcode_data.startswith("LBADVA") or
                            len(barcode_data) > 0):  # Accept any non-empty barcode
                            
                            # Add to our tracking list
                            decoded_barcodes.append((barcode_type, barcode_data))
                            
                            print(f"Barcode found with threshold {thresh_val}! Type: {barcode_type}, Data: {barcode_data}")
                            
                            # Add text at the top of the image
                            cv2.putText(image, f"{barcode_type}: {barcode_data}", 
                                      (10, 30 + 30 * len(decoded_barcodes)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error with zxing on binary image: {e}")
                
            # Try with pyzbar as a fallback
            try:
                pyzbar_results = decode(binary)
                if pyzbar_results:
                    for pyzbar_result in pyzbar_results:
                        barcode_data = pyzbar_result.data.decode('utf-8')
                        barcode_type = pyzbar_result.type
                        
                        # Add to our tracking list
                        decoded_barcodes.append((barcode_type, barcode_data))
                        
                        print(f"Barcode found with pyzbar and threshold {thresh_val}! Type: {barcode_type}, Data: {barcode_data}")
                        
                        # Add text at the top of the image
                        cv2.putText(image, f"{barcode_type}: {barcode_data}", 
                                  (10, 30 + 30 * len(decoded_barcodes)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as pyzbar_error:
                print(f"Error with pyzbar on binary image: {pyzbar_error}")
    
    # Display a summary of all decoded barcodes at the top of the image
    if decoded_barcodes:
        # Count occurrences of each barcode
        barcode_counts = {}
        for barcode_type, barcode_data in decoded_barcodes:
            key = f"{barcode_type}:{barcode_data}"
            if key in barcode_counts:
                barcode_counts[key] += 1
            else:
                barcode_counts[key] = 1
        
        # Display summary at the top of the image
        y_offset = 30
        for i, (key, count) in enumerate(barcode_counts.items()):
            barcode_type, barcode_data = key.split(":", 1)
            summary_text = f"Found {count}x {barcode_type}: {barcode_data}"
            cv2.putText(image, summary_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
    
    return image

# Initialize camera
camera = CameraStream([1706, 960], 30)
time.sleep(2)  # Allow time to start streaming

while True:
    command = input("Enter 'grab' to get a frame, 'video' for video, 'qr' for barcode scanning, 'ocr' for ocrs or 'exit' to quit: ")
    if command == 'grab':
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xff
            if key == ord('s'):
                print("Image capture requested but saving is disabled")
            cv2.destroyAllWindows()
        else:
            print("No frame available.")
    
    elif command == 'video':
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(2) & 0xff
                if key == ord('s'):
                    print("Image capture requested but saving is disabled")
                elif key == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                print("No frame available.")
    
    elif command == 'qr':
        while True:
            frame = camera.get_frame()
            if frame is not None:
                key = cv2.waitKey(2) & 0xff
                if key == ord('s'):
                    print("Image capture requested but saving is disabled")
                frame = barcode_reader(frame)
                cv2.imshow('QR Detection', frame)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                print("No frame available.")
    elif command == 'ocr':
        frame = camera.get_frame()
        if frame is not None:
            try:
                # Try to use the frame directly
                data = ocr.read(frame)
            except Exception as e:
                print(f"Direct frame processing failed: {e}")
                # Create a temporary file in memory
                import tempfile
                import os
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                try:
                    # Save frame to temporary file
                    cv2.imwrite(temp_filename, frame)
                    
                    # Process with OCR
                    data = ocr.read(temp_filename)
                    
                    # Display results
                    for item, value in data.items():
                        cv2.rectangle(frame, (int(value[0][0][0]), int(value[0][0][1])), 
                                     (int(value[0][2][0]), int(value[0][2][1])), (0, 255, 0), 2)
                    
                    cv2.imshow('OCR Detection', frame)
                    cv2.waitKey(0)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
            else:
                # If direct processing worked, display results
                for item, value in data.items():
                    cv2.rectangle(frame, (int(value[0][0][0]), int(value[0][0][1])), 
                                 (int(value[0][2][0]), int(value[0][2][1])), (0, 255, 0), 2)
                
                cv2.imshow('OCR Detection', frame)
                cv2.waitKey(0)
            
            cv2.destroyAllWindows()
        else:
            print("No frame available.")

    elif command == 'exit':
        camera.stop()
        cv2.destroyAllWindows()
        break
    else:
        print("Invalid command.")