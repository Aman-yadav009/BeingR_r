def process_image(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)
	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	for label in np.unique(labels):
		if label == 0:
			continue
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 300:
			mask = cv2.add(mask, labelMask)
	contour = cv2.findContours(mask, cv2.RETR_TREE,   cv2.CHAIN_APPROX_SIMPLE)
	contour = imutils.grab_contours(contour)
	areas = []

	for cnt in contour:
		areas.append(cv2.contourArea(cnt))
	full_areas = np.sum(areas)
	return full_areas

def test():
	video_capture = cv2.VideoCapture(0)
	retn = True
	while retn:
		ret, image = video_capture.read()
		save = image
		retn = False
	video_capture.release()	
	return save

def process_surrounding():
	max = area_list[0]
	dir_counter = 0
	direc = 0
	for i in range(len(area_list)):
		if(area_list[i] > max):
			dir_counter = i
			max = area_list[i]
	if dir_counter == 0 :
		direc = 0
	elif dir_counter == 1 :
		direc = 1
	elif dir_counter == 2 :
		dir = 2
	else:
		direc = 3
	return direc

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		print('[%s] => %d' % (value, i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[0]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	if(float(len(numbers)-1) <= 0):
		return 0
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[0])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	if(stdev == 0):
		return 0
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

def process_image_live(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)
	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	for label in np.unique(labels):
		if label == 0:
			continue
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 300:
			mask = cv2.add(mask, labelMask)
	contour = cv2.findContours(mask, cv2.RETR_TREE,   cv2.CHAIN_APPROX_SIMPLE)
	contour = imutils.grab_contours(contour)
	areas = []
	centersX = []
	centersY = []

	for cnt in contour:
		areas.append(cv2.contourArea(cnt))
		M = cv2.moments(cnt)
		centersX.append(int(M["m10"] / M["m00"]))
		centersY.append(int(M["m01"] / M["m00"]))
	full_areas = np.sum(areas)
	acc_X = 0
	acc_Y = 0
	for i in range(len(areas)):
		acc_X += centersX[i] * (areas[i]/full_areas)
		acc_Y += centersY[i] * (areas[i]/full_areas)
	#dimensions = image.shape
	height = image.shape[0] / 2
	width = image.shape[1] / 2
	#print (acc_X, acc_Y)
	cv2.circle(image, (int(width), int(height)), 5, (255,0,0), -1)
	cv2.circle(image, (int(acc_X), int(acc_Y)), 5, (255, 0, 0), -1)
	if(acc_X < width - 10):
		return 1
	elif(acc_X > width + 10):
		return 2
	else:
		return 0 

def rr90():
	print("rotate right by 90")
	arduino.write('r'.encode())
	time.sleep(2)

def left():
	print("left")
	arduino.write('u'.encode())
	time.sleep(2)

def right():
	print("right")
	arduino.write('t'.encode())
	time.sleep(2)

def lr90():
	print("rotate left by 90")
	arduino.write('l'.encode())
	time.sleep(2)

def r180():
	print("rotate by 180")
	arduino.write('b'.encode())
	time.sleep(2)

def forward():
	print("forwarded")
	arduino.write('f'.encode())
	time.sleep(2)

def read_weather():
	arduino.write('w'.encode())
	#time.sleep(1)
	arduino.readline()
	string_n = arduino.decode()
	string = string_n.rstrip()
	values = string.split()
	val_list = []
	for i in range(len(values)):
		val_list.append(float(values[i]))
	t_w = []
	t_w.append(val_list[2])
	t_w.append(val_list[1])
	t_w.append(val_list[0])
	return t_w

def read_light():
	threshold = 0
	arduino.write("2")
	time.sleep(1)
	arduino.readline()
	string_n = arduino.decode()
	string = string_n.rstrip()
	value = float(string)
	if value < threshold:
		return True
	else:
		return False

def direction_stack(count, direction):
	dir_stack.append(direction)
	count_stack.append(count)

def init_stack():
	dir_stack.append('x')
	count_stack.append(0)