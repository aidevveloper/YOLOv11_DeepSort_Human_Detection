import warnings

import cv2
import numpy
import torch

from nets import nn
from utils import util

warnings.filterwarnings("ignore")


def draw_line(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (200, 0, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f'ID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 2),
                0, 1 / 2, (0, 255, 0),
                thickness=1, lineType=cv2.FILLED)


def main():
    size = 640
    model = torch.load('weights/best.pt', map_location='cuda')['model'].float()
    model.eval()
    model.half()
    
    # Warmup model
    dummy = torch.zeros((1, 3, size, size), device='cuda', dtype=torch.float16)
    with torch.no_grad():
        _ = model(dummy)
    del dummy
    
    reader = cv2.VideoCapture('assets/people.mp4')
    deepsort = nn.DeepSort()

    # Check if camera opened successfully
    if not reader.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while reader.isOpened():
        # Capture frame-by-frame
        success, frame = reader.read()
        if success:
            shape = frame.shape[:2]

            r = size / max(shape[0], shape[1])
            if r != 1:
                h, w = shape
                image = cv2.resize(frame,
                                   dsize=(int(w * r), int(h * r)),
                                   interpolation=cv2.INTER_LINEAR)
            else:
                image = frame

            h, w = image.shape[:2]
            image, ratio, pad = util.resize(image, size)
            shapes = shape, ((h / shape[0], w / shape[1]), pad)
            
            # Optimized conversion
            sample = torch.from_numpy(numpy.ascontiguousarray(image.transpose((2, 0, 1))[::-1])).unsqueeze(0)
            sample = sample.to('cuda', dtype=torch.float16, non_blocking=True) / 255.0

            # Inference
            with torch.no_grad():
                outputs = model(sample)

            # NMS
            outputs = util.non_max_suppression(outputs, 0.35, 0.45)
            
            boxes = []
            confidences = []
            object_indices = []
            
            for i, output in enumerate(outputs):
                if len(output) == 0:
                    continue
                detections = output.clone()
                util.scale(detections[:, :4], sample[i].shape[1:], shapes[0], shapes[1])
                detections = detections.cpu().numpy()
                detections = util.xy2wh(detections)
                
                boxes.extend(detections[:, :4].astype(int).tolist())
                confidences.extend(detections[:, 4:5].tolist())
                object_indices.extend([0] * len(detections))
            
            if len(boxes) > 0:
                boxes_array = numpy.array(boxes, dtype=numpy.float32)
                confidences_array = numpy.array(confidences, dtype=numpy.float32)
                outputs = deepsort.update(boxes_array, confidences_array, object_indices, frame)
            else:
                outputs = []

            if len(outputs) > 0:
                boxes = outputs[:, :4].astype(int)
                object_id = outputs[:, -1]
                identities = outputs[:, -2].astype(int)
                for i, (box, obj_id, identity) in enumerate(zip(boxes, object_id, identities)):
                    if obj_id != 0:  # 0 is for person class (COCO)
                        continue
                    x1, y1, x2, y2 = box
                    draw_line(frame, x1, y1, x2, y2, identity)
                    
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()