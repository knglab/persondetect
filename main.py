import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import argparse
import tensorflow as tf
import utils
# Teminal arguments
parser = argparse.ArgumentParser(description='YoloV5 detection')
parser.add_argument('-i', '--input', type=str,
                    help='Input image .jpg or png',required=True)
parser.add_argument('-m', '--model', type=str,
                    help='Path to model tflite file', default='./pretrained/yolov5m-fp16.tflite')
args = parser.parse_args()

def main(args):
    # Load input image
    img_raw = cv2.imread(args.input)
    input_data,ratio,padding = utils.prepare_image(img_raw)
    # Loading model
    # Create the interpreter and signature runner
    interpreter = tf.lite.Interpreter(args.model)
    interpreter.allocate_tensors()  # allocate
    input_details = interpreter.get_input_details()  # inputs
    output_details = interpreter.get_output_details()  # outputs
    
    # Inferencing
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predictions = utils.decode_prediction(output_data)
    predictions[:,:4] = utils.scale_coords(input_data.shape[1:3],predictions[:,:4],ratio,padding)
    predictions[:,:4] = utils.clip_coords(predictions[:,:4],img_raw.shape[:2])
    
    for pred in predictions:
    	if utils.int2str(int(pred[-1])) == 'person':
            color = (0,255,0)
            cv2.rectangle(img_raw, (int(pred[0]), int(pred[1])), 
                                    (int(pred[2]), int(pred[3])), color, 2)
            cv2.putText(img_raw,"{}:{:.2f}".format("person",pred[4]), (int(pred[0]), int(pred[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("output.jpg", img_raw)
    cv2.imshow("output", img_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main(args)
