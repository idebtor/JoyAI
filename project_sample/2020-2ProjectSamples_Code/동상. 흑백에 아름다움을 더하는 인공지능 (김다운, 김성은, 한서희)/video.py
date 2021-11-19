import cv2 # opencv 3.4.2+ required
import numpy as np
import sys

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  exit()

# initialize writing video
output_size = (
  640,
  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 640 / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

proto = './models/colorization_deploy_v2.prototxt'
weights = './models/colorization_release_v2.caffemodel'
# weights = './models/colorization_release_v2_norebal.caffemodel'
# load cluster centers
pts_in_hull = np.load('./models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

# load model
net = cv2.dnn.readNetFromCaffe(proto, weights)
# net.getLayerNames()

# populate cluster centers as 1x1 convolution kernel
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

skipping = False

while True:
  ret, img = cap.read()

  if not ret:
    break

  img = cv2.resize(img, output_size)

  pred_bgr = img.copy()
  img_ori = img.copy()

  # normalize input
  img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
  # img_ori = cv2.equalizeHist(img_ori)
  # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
  # img_ori = clahe.apply(img_ori)
  img_ori = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2RGB)

  if not skipping:
    img_ori = (img_ori / 255.).astype(np.float32)

    # convert RGB to LAB
    img_lab = cv2.cvtColor(img_ori, cv2.COLOR_RGB2Lab)
    # only L channel to be used
    img_l = img_lab[:, :, 0]

    input_img = cv2.resize(img_l, (224, 224))
    input_img -= 50 # subtract 50 for mean-centering

    # prediction
    net.setInput(cv2.dnn.blobFromImage(input_img))

    pred = net.forward()[0,:,:,:].transpose((1, 2, 0))

    # resize to original image shape
    pred_resize = cv2.resize(pred, (img.shape[1], img.shape[0]))

    # concatenate with original image L
    pred_lab = np.concatenate([img_l[:, :, np.newaxis], pred_resize], axis=2)

    # convert LAB to RGB
    pred_bgr = cv2.cvtColor(pred_lab, cv2.COLOR_Lab2BGR)
    pred_bgr = np.clip(pred_bgr, 0, 1) * 255
    pred_bgr = pred_bgr.astype(np.uint8)

  # visualize
  cv2.imshow('img_ori', img_ori)
  cv2.imshow('pred_bgr', pred_bgr)
  out.write(pred_bgr)

  key = cv2.waitKey(1)
  if key == ord('q'):
    break
  elif key == ord('s'):
    skipping = not skipping

  print('%s/%s' % (cap.get(cv2.CAP_PROP_POS_FRAMES), n_frames), end='\r')

cap.release()
out.release()
cv2.destroyAllWindows()
