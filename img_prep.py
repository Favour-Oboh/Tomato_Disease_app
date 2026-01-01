import cv2
import numpy as np
import torch

img_size = 300

def img_process(image_file):
  file_bytes = np.asarray(
      bytearray(image_file.read()), dtype=np.uint8
  )

  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  img = cv2.cvtColor(
      cv2.resize(img, (img_size, img_size)),
      cv2.COLOR_BGR2RGB
  )

  img = img.astype(np.float32) / 255.0

  tensor = torch.from_numpy(
      img.transpose(2, 0, 1)
  ).unsqueeze(0)

  return tensor
