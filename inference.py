import cv2
import numpy as np
import argparse
import qcsnpe as qc


INPUT_WIDTH = 640
INPUT_HEIGHT = 384
CPU = 0
GPU = 1
DSP = 2

model_path = "model_data/HybridNets_384x640.dlc"
out_layers = ["/classifier/Sigmoid","/regressor/Concat_5", "/segmentation_head/segmentation_head.1/Resize" ]
model = qc.qcsnpe(model_path,out_layers, CPU)
segmentation_colors = np.array([[0,    0,    0],
								[34,   139, 34],
								[0,    0,  255]], dtype=np.uint8)

def prepare_input(image):

	img_height, img_width = image.shape[:2]
	input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Resize input image
	input_img = cv2.resize(input_img, (INPUT_WIDTH ,INPUT_HEIGHT)) 

	return input_img

def draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors
	color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
	color_segmap[seg_map>0] = segmentation_colors[seg_map[seg_map>0]]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return combined_img

def hybridnet():
	# Initialize video
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--vid", default=None,help="cam/video_path")
	args = vars(ap.parse_args())
	vid = args.get("vid")
	if vid is None:
		print("Required command line args atleast ----img_folder <image folder path> or --vid <cam/video_path>")
		exit(0)
	if vid == "cam":
		video_capture = cv2.VideoCapture("tcp://localhost:8080") #RB5 Gstreamer input
	else:
		video_capture = cv2.VideoCapture(vid)

	start_time = 0 
	video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out_video = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))

	while video_capture.isOpened():
		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break
		try:
			# Read frame from the video
			ret, new_frame = video_capture.read()
			if not ret:	
				break
		except:
			continue
		#Pre-processing
		frame = new_frame.copy()
		new_frame = prepare_input(new_frame)
		#Video Inferencing
		out = model.predict(new_frame)
		#Post-processing
		seg_map = out.get("segmentation")
		if seg_map is None:
			print("Error occured during processing output.")
			exit(0)
		seg_map = np.array(seg_map)
		seg_map = np.reshape(seg_map,(1,INPUT_HEIGHT,INPUT_WIDTH,3))
		seg_map = np.squeeze(np.argmax(seg_map, axis=3))
		seg_img = draw_seg(seg_map, frame, 0.5)
		
		frame_c = cv2.resize(seg_img,(640, 480))
		cv2.imwrite("images/output.jpg", frame_c)
		out_video.write(frame_c)

		video_capture.release()
		out_video.release()

if __name__ == "__main__":
	hybridnet()
