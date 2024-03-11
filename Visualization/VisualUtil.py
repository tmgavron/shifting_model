import math
from PIL import Image, ImageDraw, ImageFont
import configparser
import pandas as pd
import numpy as np
import cairo

# Color Constants as RGBA tuples:
NAVY_DARK    = (0.05, 0.14, 0.25, 1.00) # Auburn Navy Dark
NAVY_LIGHT   = (0.81, 0.83, 0.85, 1.00) # Auburn Navy Light
ORANGE_DARK  = (0.91, 0.38, 0.00, 1.00) # Auburn Orange Dark
ORANGE_LIGHT = (0.98, 0.87, 0.80, 1.00) # Auburn Orange Light
WHITE        = (1.00, 1.00, 1.00, 1.00) # White
BLACK        = (0.00, 0.00, 0.00, 1.00) # Black
TRANSPARENT  = (0.00, 0.00, 0.00, 0.00) # Transparent

# Global Variable Declarations
DISTANCE_TO_PLATE  =  60.5
DISTANCE_TO_GRASS  =  95.0
DISTANCE_TO_FENCE  = 339.5
image_scale_factor =   2
cushion            =  10   # Extra space around the field to allow for thicker lines / outlines. Uniform on all sides.

config = configparser.ConfigParser()
config.read('Data//config.ini')

# Creates an image of a baseball field with slices colored based on the odds of being hit into. Also displays the percent chance on each slice.

# Eventually:
# int(config['VISUAL']['InfieldSlices'])
# int(config['VISUAL']['OutfieldSlices'])
def visualizeData(infieldPercentages, outfieldPercentages, filename):
    initializeFieldVariables()
    infield_slices  = infieldPercentages.__len__()
    outfield_slices = outfieldPercentages.__len__()

    # Create the field image
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, FIELD_WIDTH, FIELD_HEIGHT)
    draw = cairo.Context(surface)

    # Draw the slices & layer field lines on top
    if(config['VISUAL']['RenderOutfield']=='True'):
        fillSlices(draw, outfield_slices, outfieldPercentages, DISTANCE_TO_FENCE, OUTFIELD_ARC, NAVY_LIGHT,   NAVY_DARK)
        fillSlices(draw, infield_slices,  infieldPercentages,  DISTANCE_TO_GRASS, INFIELD_ARC,  ORANGE_LIGHT, ORANGE_DARK)
        drawField(draw, infield_slices, outfield_slices)
    else:
        fillSlices(draw, infield_slices,  infieldPercentages,  DISTANCE_TO_FENCE, OUTFIELD_ARC,  ORANGE_LIGHT, ORANGE_DARK)
        drawOnlyInfield(draw, infield_slices)

    surface.write_to_png('Visualization/' + filename + '.png')

    # Write text on top of the image
    image = Image.open('Visualization/' + filename + '.png')
    if(config['VISUAL']['RenderOutfield']=='True'):
        image = addPercents(image, infield_slices,  infieldPercentages,  DISTANCE_TO_GRASS)
        image = addPercents(image, outfield_slices, outfieldPercentages, DISTANCE_TO_FENCE)
    else:
        image = addPercents(image, infield_slices,  infieldPercentages,  DISTANCE_TO_FENCE)

    image.save('Visualization/' + filename + '.png')


# Top left corner is (0,0)
# Bottom right corner is (field_width, field_height)
def initializeFieldVariables():
    global DISTANCE_TO_PLATE, DISTANCE_TO_GRASS, DISTANCE_TO_FENCE, PLATE, MOUND, FOULL, FOULR, OUTFIELD_ARC, INFIELD_ARC, FIELD_HEIGHT, FIELD_WIDTH
    DISTANCE_TO_PLATE *= image_scale_factor
    DISTANCE_TO_GRASS *= image_scale_factor
    DISTANCE_TO_FENCE *= image_scale_factor

    foul_line_intercept = getIntersection((0, DISTANCE_TO_PLATE, DISTANCE_TO_FENCE), (0,0), math.radians(45))
    base_line_intercept = getIntersection((0, DISTANCE_TO_PLATE, DISTANCE_TO_GRASS), (0,0), math.radians(45))
    FIELD_HEIGHT = int(math.ceil((DISTANCE_TO_PLATE + DISTANCE_TO_FENCE)) + (2*cushion))
    FIELD_WIDTH  = int(math.ceil(foul_line_intercept[0] * 2) + (2*cushion))
    base_width   = int(math.ceil(base_line_intercept[0] * 2))
    #print('Field Height: ' + str(field_height) + '\nField Width: ' + str(field_width))

    PLATE = (FIELD_WIDTH/2, FIELD_HEIGHT - cushion)
    MOUND = (FIELD_WIDTH/2, FIELD_HEIGHT - cushion - DISTANCE_TO_PLATE)
    FOULL = (cushion, FIELD_HEIGHT - cushion - math.ceil(foul_line_intercept[1]))
    FOULR = (FIELD_WIDTH - cushion, FIELD_HEIGHT - cushion - math.ceil(foul_line_intercept[1]))
    OUTFIELD_ARC = math.asin((FIELD_WIDTH - 2*cushion)/(2*DISTANCE_TO_FENCE))
    INFIELD_ARC  = math.asin((base_width)/(2*DISTANCE_TO_GRASS))

def drawField(draw, infield, outfield):
    drawFieldLines(draw, 12, WHITE)
    drawInfieldSliceLines (draw, 12, WHITE, infield)
    drawOutfieldSliceLines(draw, 12, WHITE, outfield)
    drawFieldSplit(draw, 12, WHITE)
    drawFieldSplit(draw,  6, BLACK)
    drawFieldLines(draw,  6, BLACK)
    drawInfieldSliceLines (draw,  6, BLACK, infield)
    drawOutfieldSliceLines(draw,  6, BLACK, outfield)

def drawOnlyInfield(draw, infield):
    drawFieldLines(draw, 12, WHITE)
    drawInfieldSliceLines (draw, 12, WHITE, infield)
    drawOutfieldSliceLines(draw, 12, WHITE, infield)
    drawFieldLines(draw,  6, BLACK)
    drawInfieldSliceLines (draw,  6, BLACK, infield)
    drawOutfieldSliceLines(draw,  6, BLACK, infield)

def drawFieldLines(draw, thick, color):
    draw.move_to(PLATE[0], PLATE[1])
    draw.arc(MOUND[0], MOUND[1], DISTANCE_TO_FENCE, -math.pi/2 - OUTFIELD_ARC, -math.pi/2 + OUTFIELD_ARC)
    draw.close_path()
    draw.set_line_width(thick)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.stroke()

def drawFieldSplit(draw, thick, color):
    draw.arc(MOUND[0], MOUND[1], DISTANCE_TO_GRASS, -math.pi/2 - INFIELD_ARC, -math.pi/2 + INFIELD_ARC)
    draw.set_line_width(thick)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.stroke()

def drawInfieldSliceLines(draw, thick, color, slices):
    for i in range(1, slices):
        start = (PLATE[0], PLATE[1])
        end   = flip_y(getIntersection(flip_y((MOUND[0], MOUND[1], DISTANCE_TO_GRASS)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + (i * 90 / slices))))
        drawSliceLine(draw, start, end, thick, color)

def drawOutfieldSliceLines(draw, thick, color, slices):
    for i in range(1, slices):
        start  = flip_y(getIntersection(flip_y((MOUND[0], MOUND[1], DISTANCE_TO_GRASS)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + (i * 90 / slices))))
        end    = flip_y(getIntersection(flip_y((MOUND[0], MOUND[1], DISTANCE_TO_FENCE)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + (i * 90 / slices))))
        drawSliceLine(draw, start, end, thick, color)

def drawSliceLine(draw, start, end, thick, color):
    draw.move_to(start[0], start[1])
    draw.line_to(end[0], end[1])
    draw.set_line_width(thick)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.stroke()

def fillSlices(draw, slices, percentages, arc_distance, arc_angle, color1, color2):
    maxOdds = max(percentages)
    angle_to = -math.pi/2 - arc_angle
    angle_diff = 2 * arc_angle / slices
    for i in range(0, slices):
        angle_from = angle_to
        angle_to += angle_diff
        sliceColor = blendColors(color1, color2, percentages[i]/maxOdds)
        drawFilledSlice(draw, angle_from, angle_to, sliceColor, arc_distance)
    
def drawFilledSlice(draw, angle_from, angle_to, color, radius):
    draw.move_to(PLATE[0], PLATE[1])
    draw.arc(MOUND[0], MOUND[1], radius, angle_from, angle_to)
    draw.close_path()
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.fill()

def drawText(image, text, position):
    useFont = ImageFont.truetype("Visualization/Fonts/SweetSansProRegular.otf", 30)
    draw = ImageDraw.Draw(image)
    draw.text((position[0], position[1]), text, font=useFont, fill=color10to255(BLACK), align="center", anchor="mm", stroke_width=3, stroke_fill=color10to255(WHITE))

def addPercents(image, slices, percentages, arc_distance):
    for i in range(0, slices):
        pos = flip_xy(getIntersection(flip_y((MOUND[0], MOUND[1], arc_distance * 0.75)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + ((i+0.5) * 90 / slices))))
        drawText(image, cleanNumber(percentages[i]), pos)
    return image

# Blends two RGBA colors together based on a ratio (0..1)
def blendColors(color1, color2, ratio):
    blend = [(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(4)]
    return tuple(blend)

def cleanNumber(num):
    clean = int(round(num*100))
    return str(clean) + "%"

# Circle = (x, y, r)
def getIntersection(circle, line_start, angle):
    x = line_start[0] - circle[0]
    y = line_start[1] - circle[1]
    r = circle[2]
    dirX = math.cos(angle)
    dirY = math.sin(angle)

    # Split function for debugging
    xydir = x*dirX + y*dirY
    dir2  = dirX**2 + dirY**2
    rxy   = r**2 - x**2 - y**2
    root  = xydir**2 + dir2*rxy

    t = (math.sqrt(root) - xydir) / dir2
    intersection = (x + dirX*t, y + dirY*t)
    intersection = (circle[0] + intersection[0], circle[1] + intersection[1])
    #print('X intersection: ' + str(intersection[0]) + '\nY intersection: ' + str(intersection[1]))
    return intersection

def flip_y(coords):
    if(coords.__len__() == 2):
        return (coords[0], FIELD_HEIGHT - coords[1])
    if(coords.__len__() == 3):
        return (coords[0], FIELD_HEIGHT - coords[1], coords[2])
    
def flip_x(coords):
    if(coords.__len__() == 2):
        return (FIELD_WIDTH - coords[0], coords[1])
    if(coords.__len__() == 3):
        return (FIELD_WIDTH - coords[0], coords[1], coords[2])
    
def flip_xy(coords):
    return flip_x(flip_y(coords))
    
def color10to255(color):
    return (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))
