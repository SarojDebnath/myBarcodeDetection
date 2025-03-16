import depthai as dai
import cv2
import subprocess

# Function to decode barcode using zxing-cpp
def decode_barcode_zxing(image):
    try:
        # Save the frame temporarily
        cv2.imwrite("temp_frame.png", image)
        
        # Call zxing-cpp (adjust path/command as per your setup)
        result = subprocess.run(["zxing-cpp", "temp_frame.png"], capture_output=True, text=True)
        decoded_text = result.stdout.strip()
        if decoded_text:
            return decoded_text
        else:
            return "No barcode detected"
    except Exception as e:
        return f"Error decoding barcode: {str(e)}"

# Main function to set up DepthAI and process frames
def main():
    # Create DepthAI pipeline
    pipeline = dai.Pipeline()

    # Define color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4000X3000)  # 4056x3040
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
            
            # Directly decode the entire frame using zxing-cpp
            barcode_data = decode_barcode_zxing(frame)
            print(f"Decoded: {barcode_data}")
            
            # Optional: Display the frame with decoded text for debugging
            cv2.putText(frame, barcode_data, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Barcode Detection", frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
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