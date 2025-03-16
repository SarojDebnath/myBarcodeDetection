import depthai as dai
import cv2
import zxingcpp
import numpy as np

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

    # Set to store unique barcode texts
    unique_barcodes = set()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        print("Connected to OAK-1 Max. Streaming at 4056x3040...")
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        try:
            while True:
                # Get frame from the camera
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                
                # Decode barcodes using zxingcpp
                barcodes = zxingcpp.read_barcodes(frame)
                if barcodes:
                    for barcode in barcodes:
                        if barcode.format.name == "Code128" or barcode.format.name == "DataMatrix" or barcode.format.name == "QRCode":
                            # Extract barcode info
                            decoded_text = barcode.text
                            position = barcode.position
                            # Add to unique set
                            unique_barcodes.add(decoded_text)

                            # Extract bounding box coordinates
                            top_left = (position.top_left.x, position.top_left.y)
                            bottom_right = (position.bottom_right.x, position.bottom_right.y)

                            # Draw green bounding box
                            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                            # Draw red text above the box
                            text_position = (top_left[0], top_left[1] - 10)  # Slightly above the box
                            cv2.putText(frame, decoded_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                        1.0, (0, 0, 255), 2, cv2.LINE_AA)

                # Resize frame for display to 1706x720
                display_frame = cv2.resize(frame, (1280, 720))

                # Display the resized frame
                cv2.imshow("Barcode Detection", display_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            # Print unique barcodes at the end
            if unique_barcodes:
                print("\nUnique barcodes detected:")
                for barcode in sorted(unique_barcodes):  # Sort for consistent output
                    print(f" - \"{barcode}\"")
            else:
                print("\nNo unique barcodes detected.")

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")