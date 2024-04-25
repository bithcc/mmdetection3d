categories = {
    '0:car': (100, 150, 245),  
    '1:bicycle': (100, 230, 245),  
    '2:motorcycle': (30, 60, 150),
    '3:truck': (80, 30, 180),  
    '4:other-vehicle': (100, 80, 250),  
    '5:person': (155, 30, 30),
    '6:bicyclist': (255, 40, 200),  
    '7:motorcyclist': (150, 30, 90),  
    '8:road': (255, 0, 255),
    '9:parking': (255, 150, 255),  
    '10:sidewalk': (75, 0, 75),  
    '11:other-ground': (175, 0, 75),
    '12:building': (255, 200, 0),  
    '13:fence': (255, 120, 50),  
    '14:vegetation': (0, 175, 0),
    '15:trunk': (135, 60, 0),  
    '16:terrain': (150, 240, 80),  
    '17:pole': (255, 240, 150),
    '18:traffic-sign': (255, 0, 0),  
    '19:ignore': (255,255,255)
}

# Function to convert RGB to Hex
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# Converting and printing Hex codes
hex_colors = {key: rgb_to_hex(color) for key, color in categories.items()}
print(hex_colors)
