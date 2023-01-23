import cv2


def overlay_bbox_on_image(image, bboxes):
    image_new = image.copy()
    r, c = image.shape[:2]

    for box in bboxes:
        cx = int(c * box[1])
        cy = int(r * box[2])
        w = int(c * box[3])
        h = int(r * box[4])

        x = int(cx - w/2)
        y = int(cy - h/2)
        cv2.rectangle(image_new, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_new
