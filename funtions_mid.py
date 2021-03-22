from functions import *
import joblib

def inspect_surroundings():
	img_list = []
	area_list = []
	direction = 0
	print("yo")
	area_list.append(process_image(test()))
	rr90()
	area_list.append(process_image(test()))
	rr90()
	area_list.append(process_image(test()))
	rr90()
	area_list.append(process_image(test()))
	rr90()
	direction = process_surrounding()
	if direction == 0 :
		#nothing
		print("nothing")
	elif direction == 1:
		rr90()
		direction_stack(90,'r')
	elif direction == 2 :
		lr90()
		direction_stack(90,'l')
	else :
		r180()
		direction_stack(180,'b')

def calib():
	video_capture = cv2.VideoCapture(0)
	#status = True
	while True:
		ret, image = video_capture.read()
		if (process_image_live(image) == 0):
			if l_count > 0:
				direction_stack(l_count,'l')
			else :
				direction_stack(r_count,'r')
			break
		elif (process_image_live(image) == 1):
			left()
			l_count += 1
			#status = read_light()
		elif (process_image_live(image) == 2):
			right()
			r_count += 1
			#status = read_light()
		#cv2.imshow("video", image)
		elif cv2.waitKey(1) & 0xFF == ord('q'):
			break
		else:
			print("ding ding")
	video_capture.release()

def hunt():
	status = True
	for_count = 0
	while status:
		forward()
		status = False
		for_count += 1
	direction_stack(for_count, 'f')

def homing():
	#code here
	print("nothing")
def pre_we():
	model = joblib.load('weather.pkl')
	label = predict(model, read_weather())
	return label