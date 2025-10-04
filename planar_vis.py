from PIL import Image, ImageDraw
import math


def draw_grasp_rectangle(image, label, color="green", line_width=2):
    """
    Draw rotated grasp rectangle on image using PIL
    """
    x, y, length, width, angle = label

    draw = ImageDraw.Draw(image)

    half_length = length / 2
    half_width = width / 2

    corners = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    ]

    angle_rad = math.radians(angle)
    rotated_corners = []
    for corner in corners:
        rx = corner[0] * math.cos(angle_rad) - corner[1] * math.sin(angle_rad)
        ry = corner[0] * math.sin(angle_rad) + corner[1] * math.cos(angle_rad)
        rotated_corners.append((rx + x, ry + y))

    for i in range(4):
        start = rotated_corners[i]
        end = rotated_corners[(i + 1) % 4]
        draw.line([start, end], fill=color, width=line_width)

    return image
