import cv2
import imutils
from skimage.metrics import structural_similarity

# ...a bunch of functions will be going here...
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def compare(or1, or2, im1, img2, diffs):
    (score, diff) = structural_similarity(im1, img2, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # aggregate the contours, throwing away duplicates
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        region = [x, y, x + w, y + h]
        try:
            diffs.index(region)
        except ValueError:
            diffs.append(region)

    return diffs

def filter_diffs(diffs):
    def not_contained(e, diffs):
        for t in diffs:
            if e[0] > t[0] and e[2] < t[2] and e[1] > t[1] and e[3] < t[3]:
                return False
        return True

    return [e for e in diffs if not_contained(e, diffs)]

RED = (0,0,255)

def highlight_diffs(a, b, diffs):
    diffed = b.copy()

    for area in filter_diffs(diffs):
        x1, y1, x2, y2 = area

        # is this a relocation, or an addition/deletion?
        org = find_in_original(a, b, area)
        if org is not None:
            cv2.rectangle(a, (org[0], org[1]), (org[2], org[3]), BLUE, 2)
            cv2.rectangle(diffed, (x1, y1), (x2, y2), BLUE, 2)
        else:
            cv2.rectangle(diffed, (x1+2, y1+2), (x2-2, y2-2), GREEN, 1)
            cv2.rectangle(diffed, (x1, y1), (x2, y2), RED, 2)

    cv2.imshow("Original", a)
    cv2.imshow("Diffed", diffed)
    cv2.waitKey(0)
    
def find_in_original(a, b, area):
    crop = b[area[1]:area[3], area[0]:area[2]]
    result = cv2.matchTemplate(crop, a, cv2.TM_CCOEFF_NORMED)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = maxLoc
    endX = startX + (area[2] - area[0])
    endY = startY + (area[3] - area[1])
    ocrop = a[startY:endY, startX:endX]

    # this basically needs to be a near-perfect match
    # for us to consider it a "moved" region rather than
    # a genuine difference between A and B.
    if structural_similarity(gray(ocrop), gray(crop)) >= 0.99:
        return [startX, startY, endX, endY]
diffs = compare(imageA, imageB, gray(imageA), gray(imageB), [])

if len(diffs) > 0:
    highlight_diffs(imageA, imageB, diffs)

else:
    print("no differences detected")
    
