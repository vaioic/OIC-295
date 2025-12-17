import os
import openslide

def print_properties(image_path):

    fn = os.path.basename(image_path)
    slide = openslide.OpenSlide(image_path)

    print(f"Slide: {fn}")
    print(f"Level Count: {slide.level_count}")
    print(f"Dimensions (Level 0): {slide.dimensions}") # (width, height)
    print(f"Dimensions (Levels): {slide.level_dimensions}")
    print(f"Level Downsamples: {slide.level_downsamples}")   
    print(f"Properties: {slide.properties}")
    print(f"x-scale: {slide.properties[openslide.PROPERTY_NAME_MPP_X]}")
    print(f"y-scale: {slide.properties[openslide.PROPERTY_NAME_MPP_Y]}")

    slide.close()

image_A_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\259270.svs"

image_B_path = "\\\\pn.vai.org\\projects_primary\\moore\\vari-core-generated-data\\PBC-Aperio Images\\261082.svs"

print_properties(image_A_path)
print_properties(image_B_path)

# Can you read the whole tile into memory?
slide_A = openslide.OpenSlide(image_A_path)
image_A = slide_A.read_region((0,0), 0, slide_A.dimensions)
slide_A.close()