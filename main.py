import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Args:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = os.path.join(base_dir, 'darknet', 'cfg', 'yolov3.cfg')
        self.weights = os.path.join(base_dir, 'weights', 'final_weights', 'yolov3_final.weights')
        self.classes = os.path.join(base_dir, 'darknet', 'data', 'obj.names')

args = Args()

def get_output(model):
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    return output_layers

def recognize_plate(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' not found.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from '{image_path}'.")
        return

    Height, Width, _ = image.shape
    scale_factor = 0.00392
    
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    model = cv2.dnn.readNet(args.weights, args.config)
    blob = cv2.dnn.blobFromImage(image, scale_factor, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outputs = model.forward(get_output(model))

    class_ids = []
    confidences = []
    bounding_boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                bounding_boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bounding_boxes[i]
            cropped_image = image[y:y+h, x:x+w]
            # Preprocessing for OCR
            gray_plate = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary_plate, config='--psm 7') # PSM 7 for single line of text
            print(f"Detected Plate Number: {text.strip()}")
            cv2.imshow("Cropped Plate", cropped_image)
            cv2.imshow("Processed for OCR", binary_plate)

    else:
        print("No license plate detected.")

    cv2.imshow("License Plate Detection", image)
    key = cv2.waitKey(2000)  # Waits for 2 seconds, then continues
    cv2.destroyAllWindows()

      

if __name__ == '__main__':
    try:
        image_path = input("Enter the path to the image: ").strip('"')
        recognize_plate(image_path)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()