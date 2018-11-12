import cv2
import time
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


DELAY = 0.02
USE_CAM = 1
IS_FOUND = 0
REAL_TEXT = ''

MORPH = 7
CANNY = 250

_width = 600.0
_height = 420.0
_margin = 0.0

if USE_CAM:
	video_capture = cv2.VideoCapture(0)

corners = np.array(
	[
		[[_margin, _margin]],
		[[_margin, _height + _margin]],
		[[_width + _margin, _height + _margin]],
		[[_width + _margin, _margin]],
	]
)

pts_dst = np.array(corners, np.float32)

while True:

	if USE_CAM:
		ret, rgb = video_capture.read()
	else:
		ret = 1
		rgb = cv2.imread("ocvi_1541468967.9157948_org.jpg", 1)

	if (ret):

		gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
		# gray = cv2.bilateralFilter(gray, 1, 10, 120)

		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
							cv2.THRESH_BINARY, 11, 2)

		# edges = cv2.Canny(gray, 1, CANNY)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
		closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

		img, contours, h = cv2.findContours(
		    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for cont in contours:

			if cv2.contourArea(cont) > 5000:

				arc_len = cv2.arcLength(cont, True)

				approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)

				if (len(approx) == 4):
					IS_FOUND = 1

					pts_src = np.array(approx, np.float32)

					h, status = cv2.findHomography(pts_src, pts_dst)
					out = cv2.warpPerspective(
						rgb, h, (int(_width + _margin * 2), int(_height + _margin * 2))
                    )

					# text = pytesseract.image_to_string(out)

					# if(text.strip() != ''):
					# 	REAL_TEXT = text
					# 	cv2.putText(rgb, text, (int((_width + _margin) / 2), int((_height + _margin) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

					cv2.drawContours(rgb, [approx], -1, (255, 0, 0), 2)

				else:
					pass

		# cv2.imshow( 'closed', closed )
		# cv2.imshow( 'gray', gray )
		cv2.imshow('edges', edges)
		cv2.putText(rgb, REAL_TEXT, (int((_width + _margin) / 2), int((_height + _margin) / 2)),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.imshow('original', rgb)

		if IS_FOUND:
			cv2.imshow('cut out', out)
			# print(text.strip())

		if cv2.waitKey(27) & 0xFF == ord('q'):
			break

		if cv2.waitKey(99) & 0xFF == ord('c'):
			current = str(time.time())
			cv2.imwrite('ocvi_' + current + '_edges.jpg', edges)
			cv2.imwrite('ocvi_' + current + '_gray.jpg', gray)
			cv2.imwrite('ocvi_' + current + '_org.jpg', rgb)
			cv2.imwrite('ocvi_' + current + '_out.jpg', out)
			print ("Pictures saved")
		# time.sleep(DELAY)

	else:
		print ("Stopped")
		break

if USE_CAM:
	video_capture.release()
cv2.destroyAllWindows()

# end
