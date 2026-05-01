from PIL import Image, ImageDraw

img = Image.open(r"C:\Falcker\cloud\falcker\AI\Operator Round TP6\stream_test_set_01_final_add_oil\DJI_20241230162401_0094_Z_OR-107-96-74203045d3794787983ca85682f949d5-[541-RM3].jpg")
draw = ImageDraw.Draw(img)

# Bounding box: x, y, width, height
x, y, w, h = 1300, 1500, 900, 800

draw.rectangle([x, y, x+w, y+h], outline="red", width=4)
img.save("result-02.jpg")
img.show()