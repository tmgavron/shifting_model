from PIL import Image, ImageDraw, ImageFont

# Color Constants as RGBA tuples:
navy_dark    = ( 12,  35,  64, 255) # Auburn Navy
navy_light   = (206, 211, 217, 255) # Auburn Navy Light Gradient
orange_dark  = (232, 97,    0, 255) # Auburn Orange
orange_light = (250, 223, 204, 255) # Auburn Orange Light Gradient
white        = (255, 255, 255, 255) # White
black        = (  0,   0,   0, 255) # Black
transparent  = (255, 255, 255,   0) # Transparent

# Creates an image of a baseball field with slices colored based on the odds of being hit into. Also displays the percent chance on each slice.
def visualizeData(infieldPercentages, outfieldPercentages):
    slices = infieldPercentages.__len__()
    fieldImage = Image.open('Visualization/Field2.png')
    
    sliceImages = doFieldSlices(slices, infieldPercentages,  orange_dark, orange_light,  'infield') + doFieldSlices(slices, outfieldPercentages, navy_dark, navy_light, 'outfield')
    flatImage = layerImages(fieldImage, sliceImages)
    finalImage = addPercents(flatImage, infieldPercentages, outfieldPercentages)

    finalImage.show()

def addPercents(image, infield, outfield):
    useFont = ImageFont.truetype("Visualization/Fonts/SweetSansProRegular.otf", 30)
    draw = ImageDraw.Draw(image)
    # Infield
    draw.text(( 610,920),cleanNumber(infield[0]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text(( 695,830),cleanNumber(infield[1]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text(( 814,800),cleanNumber(infield[2]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text(( 940,830),cleanNumber(infield[3]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text((1020,920),cleanNumber(infield[4]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    # Outfield
    draw.text(( 240,425),cleanNumber(outfield[0]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text(( 522,300),cleanNumber(outfield[1]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text(( 814,250),cleanNumber(outfield[2]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text((1105,300),cleanNumber(outfield[3]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    draw.text((1387,425),cleanNumber(outfield[4]),font=useFont,fill=black,align="center",anchor="mm",stroke_width=3,stroke_fill=white)
    return image

# Creates a list of images for each slice of the field
def doFieldSlices(slices, odds, color1, color2, string):
    images = []
    for i in range(1, slices+1):
        image = Image.open('Visualization/Slices/' + str(slices) + ' Slices/' + string + "_" + str(i) + '.png')
        image = colorImage(image, color1, color2, odds, i)
        images.append(image)
    return images

# Alpha composites all layers together
def layerImages(foreground, images):
    background = images[0]
    for i in range(images.__len__()-1):
        background = Image.alpha_composite(background, images[i+1])
    background = Image.alpha_composite(background, foreground)
    return background

# Changes the color of a slice based on the odds of it being hit
def colorImage(image, color1, color2, odds, index):
    maxOdds  = max(odds)
    normOdds = [x / maxOdds for x in odds]
    sliceColor = blendColors(color2, color1, normOdds[index-1])
    newImage = recolor(image, white, sliceColor)
    return newImage

# Changes the color of all pixes matching oldColor to newColor
def recolor(image, oldColor, newColor):
    data = image.getdata()
    newdata = []
    for item in data:
        if item == oldColor:
            newdata.append(newColor)
        else:
            newdata.append(item)
    image.putdata(newdata)
    return image

# Blends two RGBA colors together based on a ratio (0..1)
def blendColors(color1, color2, ratio):
    blend = [int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(4)]
    return tuple(blend)

def cleanNumber(num):
    clean = int(round(num*100))
    return str(clean) + "%"