from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io

tool = pyocr.get_available_tools()[0]
lang = tool.get_available_languages()[0]

req_image = 0
file = open("pdfToTxt.txt", "w")

for i in range(19, 421):

    print('Iter: ' + str(i-18) + ' of ' + str(421-18))

    image_pdf = Image(filename="/home/paulojeunon/Desktop/Alberta/New Folder/Product Design & development." +
                               str(i) + ".pdf", resolution=350)
    image_jpeg = image_pdf.convert('jpeg')

    img_page = Image(image=image_jpeg)
    req_image = img_page.make_blob('jpeg')

    txt = tool.image_to_string(
        PI.open(io.BytesIO(req_image)),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )
    file.write(txt)

print('end')

file.close()