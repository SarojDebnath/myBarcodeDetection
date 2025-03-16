import depthai as dai
import cv2
import zxingcpp

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
            
            # Decode barcode using zxingcpp
            barcodes = zxingcpp.read_barcodes(frame)
            if barcodes:
                for barcode in barcodes:
                    decoded_text = barcode.text
                    print(f"Found barcode:\n"
                          f" Text:    \"{decoded_text}\"\n"
                          f" Format:   {barcode.format}\n"
                          f" Content:  {barcode.content_type}\n"
                          f" Position: {barcode.position}")
                    
                    # Draw the barcode text on the frame
                    cv2.putText(frame, decoded_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                decoded_text = "No barcode detected"
                print(decoded_text)
                cv2.putText(frame, decoded_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
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