#!/usr/bin/env python3
# coding: utf-8

import sys
import argparse
import requests
from PIL import Image
import os.path
import mimetypes
import time
import random
import string
import math
from collections import deque
from array import array

# Mastodon API................................................................

class InvalidArgumentError(Exception):
    pass

class NetworkError(Exception):
    pass

class APIError(Exception):
    pass

class MastodonAPI:
    @staticmethod
    def make_request_static(url, method, access_token, parameters, files={}):
        """
        Raises: NetworkError, APIError
        """
        response = None
        headers = None
        if access_token:
            headers = {"Authorization" : "Bearer " + access_token}
        try:
            response = requests.request(
                method, url, data=parameters, headers=headers, files=files)
            response.raise_for_status()
        except (RequestException, HTTPError):
            raise NetworkError("The request was not completed.")
        try:
            result = response.json()
        except ValueError:
            raise APIError("The response recieved for the request could not "
                "be parsed to JSON. The response code was {0!s}.".format(
                    response.status_code))
        return result

    @staticmethod
    def register_app_with_oauth(
            base_url,
            client_name,
            redirect_uris="urn:ietf:wg:oauth:2.0:oob",
            scopes=["read", "write", "follow"]):
        """
        This registers a third-party application under the given client_name
        to the server at the given base_url. It should only be called once
        for any given application.
        """
        parameters = {
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "scopes": " ".join(scopes)
        }
        response = MastodonAPI.make_request_static(
            base_url + "/api/v1/apps",
            "POST",
            access_token=None,
            parameters=parameters)
        return response

    def __init__(self, base_url, client_id, client_secret):
        self.base_url = base_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

    def make_request(self, method, endpoint, parameters={}, files={}):
        """Convenience method for calling the unwieldy static method.
        Raises: APIError, NetworkError
        """
        return MastodonAPI.make_request_static(
            self.base_url + endpoint,
            method,
            self.access_token,
            parameters,
            files)

    def log_in(self, username, password, scopes=["read", "write", "follow"]):
        parameters = {
            "username": username,
            "password": password,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "password",
            "scope": " ".join(scopes),
        }
        response = self.make_request("POST", "/oauth/token", parameters)
        self.access_token = response["access_token"]
        return response["access_token"]

    def get_status(self, id):
        return self.make_request("GET", "/api/v1/statuses/{0!s}".format(id))

    def post_status(self, status, in_reply_to_id=None, media_ids=None):
        parameters = {"status": status}
        if in_reply_to_id:
            parameters["in_reply_to_id"] = in_reply_to_id
        if media_ids:
            parameters["media_ids[]"] = media_ids
        return self.make_request("POST", "/api/v1/statuses", parameters)

    def post_media(self, filename):
        if os.path.isfile(filename):
            mime_type = mimetypes.guess_type(filename)[0]
            file = open(filename, "rb")
        if mime_type == None:
            raise InvalidArgumentError(
                "The MIME type of file {} was not discernable.".format(
                    filename))
        possible_chars = string.ascii_uppercase + string.digits
        random_chars = [random.choice(possible_chars) for _ in range(10)]
        random_suffix = "".join(random_chars)
        upload_filename = "mastodonpyupload_{0!s}_{1!s}{2}".format(
            time.time(), random_suffix, mimetypes.guess_extension(mime_type))
        media_file_description = (upload_filename, file, mime_type)
        files = {"file": media_file_description}
        return self.make_request("POST", "/api/v1/media", files = files)

# This verbose print function prints only when the verbose option is set.
_v_print = None

def get_client_credentials(base_url, client_name, credential_filename):
    """
    The first time a client application calls this, it registers the
    application with the specified server and saves the client id and secret
    it recieves in a file. Then, every subsequent time it fetches those
    credentials from the file instead of re-registering every time it's
    called.
    """
    try:
        with open(credential_filename, "r") as file:
            client_id = file.readline().rstrip()
            client_secret = file.readline().rstrip()
    except FileNotFoundError as error:
        _v_print("Registering to {} as {}".format(base_url, client_name))
        response = MastodonAPI.register_app_with_oauth_app(
            base_url, client_name)
        client_id = response["client_id"]
        client_secret = response["client_secret"]
        try:
            with open(credential_filename, "w") as file:
                file.write(client_id + "\n")
                file.write(client_secret + "\n")
        except OSError as error:
            print(error)
            print(
                "{} could not be written.".format(credential_filename),
                file=sys.stderr)
    return client_id, client_secret

# Numeric Utilities...........................................................

def frange(start, stop, step):
    """float range"""
    i = start
    while i < stop:
        yield i
        i += step

def unorm_to_byte(x):
    """float x in [0, 1] to an integer [0, 255]"""
    return min(int(256 * x), 255)

def snorm_to_byte(x):
    """float x in [-1, 1] to an integer [0, 255]"""
    return min(int((x + 1) * 128), 255)

def byte_to_unorm(x):
    """integer x in [0, 255] to float [0, 1]"""
    return x / 255

def byte_to_snorm(x):
    """integer x in [0, 255] to float [-1, 1]"""
    return (x / 127.5) - 1

def lerp(v0, v1, t):
    """linearly interpolate"""
    assert 0 <= t <= 1
    return (1 - t) * v0 + t * v1

def unlerp(a, b, t):
    """inverse linear interpolation"""
    assert a <= t <= b and a != b
    return (t - a) / (b - a)

def clamp(x, low, high):
    return min(max(x, low), high)

def pack_tuple3(l):
    """Packs a sequence of values into a list of 3-tuples."""
    tuples = [None] * (len(l) // 3)
    for i in range(0, len(l) - 2, 3):
        tuples[i // 3] = (l[i], l[i + 1], l[i + 2])
    return tuples

# Colour Utilities............................................................

def rgb_lerp(a, b, t):
    assert 0 <= t <= 1
    return (
        int(lerp(a[0], b[0], t)),
        int(lerp(a[1], b[1], t)),
        int(lerp(a[2], b[2], t)))

def premultiplied_alpha_blend(a, b, t):
    assert 0 <= t <= 1
    it = (1 - t)
    return (
        a[0] + int(b[0] * it),
        a[1] + int(b[1] * it),
        a[2] + int(b[2] * it))

class Colour3:
    @staticmethod
    def to_byte3(c):
        return (unorm_to_byte(c[0]), unorm_to_byte(c[1]), unorm_to_byte(c[2]))

def hsl_to_rgb(h, s, l):
    if s == 0:
        return Colour3.to_byte3(l, l, l)
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        elif t > 1:
            t -= 1
        if t < (1 / 6):
            return p + 6 * t * (q - p)
        elif t < (1 / 2):
            return q
        elif t < (2 / 3):
            return p + 6 * ((2 / 3) - t) * (q - p)
        return p
    if l < 0.5:
        q = l * (1 + s)
    else:
        q = l + s - l * s
    p = 2 * l - q
    rgb = (
        hue_to_rgb(p, q, h + (1 / 3)),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - (1 / 3)))
    return Colour3.to_byte3(rgb)

def generate_palette(
    count, ranges, offset_angles, saturation_range, luminance_range):
    """
    -Analogous: Choose second and third ranges 0
    -Complementary: Choose the third range 0, and first offset angle 180
    -Split Complementary: Choose offset angles 180 +/- a small angle. The
    second and third ranges must be smaller than the difference between the
    two offset angles.
    -Triad: Choose offset angles 120 and 240
    """
    colours = [None] * count
    reference_angle = random.uniform(0, 360)
    for i in range(count):
        random_angle = random.random() * (ranges[0] + ranges[1] + ranges[2])
        if random_angle > ranges[0]:
            if random_angle < ranges[0] + ranges[1]:
                random_angle += offset_angles[0]
            else:
                random_angle += offset_angles[1]
        hue = ((reference_angle + random_angle) / 360) % 1
        saturation = random.uniform(saturation_range[0], saturation_range[1])
        luminance = random.uniform(luminance_range[0], luminance_range[1])
        colours[i] = hsl_to_rgb(hue, saturation, luminance)
    return colours

def generate_random_flat_palette(count):
    ranges = (90, 90, 0)
    offset_angles = (45, 45)
    which = random.randint(0, 3)
    if which == 0: # Analogous
        ranges = (ranges[0], 0, 0)
    elif which == 1: # Complementary
        ranges = (ranges[0], ranges[1], 0)
        offset_angles = (180, offset_angles[1])
    elif which == 2: # Split Complementary
        a = random.uniform(0, 30)
        offset_angles = (180 + a, 180 - a)
        width = 2 * a
        split = random.uniform(0, width)
        ranges = (ranges[0], split, random.uniform(0, width - split))
    elif which == 3: # Triad
        offset_angles = (120, 140)
    saturation_min = random.uniform(0.2, 1)
    saturation_range = (saturation_min, random.uniform(saturation_min, 1))
    luminance_min = random.uniform(0.05, 1)
    luminance_range = (luminance_min, random.uniform(luminance_min, 1))
    return generate_palette(
        count, ranges, offset_angles, saturation_range, luminance_range)

def generate_roughly_increasing_palette(count):
    """
    Random palette increasing in order of relative luminance (in HSL). This
    does not attempt to represent percieved lightness, like cube helix.
    """
    ranges = (90, 170, 80)
    offset_angles = (35, 45)
    which = random.randint(0, 3)
    if which == 0: # Analogous
        ranges = (ranges[0], 0, 0)
    elif which == 1: # Complementary
        ranges = (ranges[0], ranges[1], 0)
        offset_angles = (180, offset_angles[1])
    elif which == 2: # Split Complementary
        a = random.uniform(0, 30)
        offset_angles = (180 + a, 180 - a)
        width = 2 * a
        split = random.uniform(0, width)
        ranges = (ranges[0], split, random.uniform(0, width - split))
    elif which == 3: # Triad
        offset_angles = (120, 140)
    saturation_min = random.uniform(0.2, 1)
    saturation_range = (saturation_min, random.uniform(saturation_min, 1))
    luminance_range = (0.07, 0.93)
    colours = [None] * count
    reference_angle = random.uniform(0, 360)
    for i in range(count):
        random_angle = random.random() * (ranges[0] + ranges[1] + ranges[2])
        if random_angle > ranges[0]:
            if random_angle < ranges[0] + ranges[1]:
                random_angle += offset_angles[0]
            else:
                random_angle += offset_angles[1]
        hue = ((reference_angle + random_angle) / 360) % 1
        saturation = random.uniform(saturation_range[0], saturation_range[1])
        luminance = lerp(luminance_range[0], luminance_range[1], i / (count - 1))
        colours[i] = hsl_to_rgb(hue, saturation, luminance)
    return colours

# Cube Helix Palette Generation...............................................

def cube_helix(
    levels, start_hue, rotations,
    saturation_range=(0, 1), lightness_range=(0, 1), gamma=1):
    """
    Based on Dave Green's public domain (Unlicense license) Fortran 77
    implementation for cube helix colour table generation.
    """
    low = 0
    high = 0
    colours = [None] * levels
    for i in range(levels):
        fraction = lerp(lightness_range[0], lightness_range[1], i / levels)
        saturation = lerp(saturation_range[0], saturation_range[1], fraction)
        angle = TAU * (start_hue / 3 + 1 + rotations * fraction)
        fraction = math.pow(fraction, gamma)
        amplitude = saturation * fraction * (1 - fraction) / 2
        r = -0.14861 * math.cos(angle) + 1.78277 * math.sin(angle)
        g = -0.29227 * math.cos(angle) - 0.90649 * math.sin(angle)
        b = 1.97294 * math.cos(angle)
        r = fraction + amplitude * r
        g = fraction + amplitude * g
        b = fraction + amplitude * b
        if r < 0:
            r = 0
            low += 1
        if g < 0:
            g = 0
            low += 1
        if b < 0:
            b = 0
            low += 1
        if r > 1:
            r = 1
            high += 1
        if g > 1:
            g = 1
            high += 1
        if b > 1:
            b = 1
            high += 1
        colours[i] = Colour3.to_byte3((r, g, b))
    return colours, low, high

def fetch_colour(colours, value, low, high):
	u = unlerp(low, high, value)
	v = lerp(0, len(colours), u)
	v_truncated = math.floor(v)
	t = v - v_truncated
	index = int(v_truncated)
	c0 = colours[index]
	c1 = colours[index + 1]
	return rgb_lerp(c0, c1, t)

# Geometry Utilities..........................................................

class Vector2:
    @staticmethod
    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])

    @staticmethod
    def subtract(a, b):
        return (a[0] - b[0], a[1] - b[1])

    @staticmethod
    def multiply(c, v):
        return (c * v[0], c * v[1])

    @staticmethod
    def divide(v, c):
        return (v[0] / c, v[1] / c)

    @staticmethod
    def length(v):
        return math.sqrt((v[0] ** 2) + (v[1] ** 2))

    @staticmethod
    def distance(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.sqrt((dx * dx) + (dy * dy))

    @staticmethod
    def scale(c, v):
        return (c * v[0], c * v[1])

    @staticmethod
    def perp(v):
        return (v[1], -v[0])

    @staticmethod
    def floor(v):
        return (int(v[0]), int(v[1]))

    @staticmethod
    def dot(a, b):
        return (a[0] * b[0]) + (a[1] * b[1])

    @staticmethod
    def lerp(a, b, t):
        return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))

    @staticmethod
    def limit_length(v, limit):
        d = Vector2.length(v)
        if d > limit:
            return Vector2.scale(limit / d, v)
        else:
            return v

def get_circle_bounds(center, radius):
    left   = int(center[0] - radius)
    right  = int(center[0] + radius + 1)
    top    = int(center[1] - radius)
    bottom = int(center[1] + radius + 1)
    return (left, top, right, bottom)

def point_in_box(point, box):
    return (box[0] <= point[0] <= box[2] - 1
        and box[1] <= point[1] <= box[3] - 1)

def clip_box(box, bounds):
    return (
        max(box[0], bounds[0]),
        max(box[1], bounds[1]),
        min(box[2], bounds[2]),
        min(box[3], bounds[3]))

def clip_point(point, box):
    return (
        min(max(point[0], box[0]), box[2] - 1),
        min(max(point[1], box[1]), box[3] - 1))

# Canvas......................................................................

def unpack_rg(rg):
    return ((rg) & 0xFF, (rg >> 8) & 0xFF)

def unpack_rgb(rgb):
    return (rgb & 0xFF, (rgb >> 8) & 0xFF, (rgb >> 16) & 0xFF)

def unpack_rgba(rgba):
    return (
        (rgba      ) & 0xFF,
        (rgba >>  8) & 0xFF,
        (rgba >> 16) & 0xFF,
        (rgba >> 24) & 0xFF)

def pack_rg(rg):
    return (rg[1] << 8) | (rg[0])

def pack_rgb(rgb):
    return (rgb[2] << 16) | (rgb[1] << 8) | (rgb[0])

def pack_rgba(rgba):
    return (rgba[3] << 24) | (rgba[2] << 16) | (rgba[1] << 8) | (rgba[0])

# There doesn't seem to be any way to query the size in bytes of array
# typecodes. So, for each of these make_array functions, make a zero-item
# array and take its itemsize, then make the real array.

def make_array8(count):
    byte_size = array('B').itemsize
    return array('B', bytearray(byte_size * count))

def make_array16(count):
    short_size = array('H').itemsize
    return array('H', bytearray(short_size * count))

def make_array32(count):
    int_size = array('I').itemsize
    long_size = array('L').itemsize
    if int_size == 4:
        return array('I', bytearray(int_size * count))
    else:
        return array('L', bytearray(int_size * count))

def make_array_float(count):
    double_size = array('d').itemsize
    return array('d', bytearray(double_size * count))

class Canvas:
    """
    Attrs:
        pixels = an array of unsigned 1, 2, or 4 byte values
        size = a tuple containing the dimensions in pixels (width, height)
        mode = one of the strings "R", "RG", "RGB", "RGBA", "F"
    """
    def __init__(self, mode, size, pixels=None):
        if mode not in ["R", "RG", "RGB", "RGBA", "F"]:
            raise ValueError(
                """mode must be one of the strings {"R", "RG", "RGB",\
                "RGBA", "F"}.""")
        if pixels is not None:
            self.pixels = pixels
        else:
            count = size[0] * size[1]
            if mode == "R":
                self.pixels = make_array8(count)
            elif mode == "RG":
                self.pixels = make_array16(count)
            elif mode == "RGB" or mode == "RGBA":
                self.pixels = make_array32(count)
            elif mode == "F":
                self.pixels = make_array_float(count)
        self.size = size
        self.mode = mode

    def get_bounds(self):
        return (0, 0, self.size[0], self.size[1])
        
    def get_pixel_r(self, point):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        return self.pixels[index]

    def get_pixel_rg(self, point):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        return unpack_rg(self.pixels[index])

    def get_pixel_rgb(self, point):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        return unpack_rgb(self.pixels[index])

    def get_pixel_rgba(self, point):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        return unpack_rgba(self.pixels[index])

    def get_pixel_f(self, point):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        return self.pixels[index]

    def put_pixel_r(self, point, pixel):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        self.pixels[index] = pixel

    def put_pixel_rg(self, point, pixel):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        self.pixels[index] = pack_rg(pixel)

    def put_pixel_rgb(self, point, pixel):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        self.pixels[index] = pack_rgb(pixel)

    def put_pixel_rgba(self, point, pixel):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        self.pixels[index] = pack_rgba(pixel)

    def put_pixel_f(self, point, pixel):
        assert point_in_box(point, self.get_bounds())
        index = self.size[0] * point[1] + point[0]
        self.pixels[index] = pixel

    def put_and_premultiply_alpha(self, alpha):
        """Add an alpha channel to an RGB image."""
        assert self.size == alpha.size
        assert self.mode == "RGB"
        for i in range(len(self.pixels)):
            rgb = unpack_rgb(self.pixels[i])
            a_byte = alpha.pixels[i]
            a = byte_to_unorm(a_byte)
            rgb = (
                int(a * rgb[0]),
                int(a * rgb[1]),
                int(a * rgb[2]))
            self.pixels[i] = (a_byte << 24) | pack_rgb(rgb)
        self.mode = "RGBA"

    def convert(self, mode):
        """
        The implemented conversions are:
            -float (F) to 1-channel (R)
            -1-channel (R) to 3-channel (RGB)
        """
        count = len(self.pixels)
        if mode == "R":
            if self.mode == "F":
                pixels = array('B', bytearray(count))
                for i in range(count):
                    pixels[i] = clamp(int(self.pixels[i]), 0, 255)
            else:
                raise NotImplementedError(
                    "An F mode image can only be converted to an image of "
                    "mode R.")
        elif mode == "RGB":
            if self.mode == "R":
                pixels = make_array32(count)
                for i in range(count):
                    r = self.pixels[i]
                    pixels[i] = (r << 16) | (r << 8) | (r)
            elif self.mode == "RG":
                pixels = make_array16(count)
                for i in range(count):
                    pixels[i] = self.pixels[i]
            elif self.mode == "RGB":
                return self.crop(self.get_bounds())
            else:
                raise NotImplementedError(
                    "An RGB mode image can only be converted to an image of "
                    "mode R or RG.")
        return Canvas(mode=mode, size=self.size, pixels=pixels)

    def crop(self, box):
        """Only RGB and RGBA cropping is implemented!"""
        if self.mode not in ["RGB", "RGBA"]:
            raise NotImplementedError(
                "Only images of mode RGB and RGBA can be cropped.")
        box = clip_box(box, self.get_bounds())
        width  = box[2] - box[0]
        height = box[3] - box[1]
        copied_pixels = make_array32(width * height)
        i = 0
        for y in range(box[1], box[3]):
            for x in range(box[0], box[2]):
                copied_pixels[i] = self.pixels[self.size[0] * y + x]
                i += 1
        return Canvas(
            mode=self.mode, size=(width, height), pixels=copied_pixels)

    def copy(self, image, box):
        box = clip_box(box, self.get_bounds())
        for y in range(box[1], box[3]):
            for x in range(box[0], box[2]):
                si = self.size[0] * y + x
                ci = image.size[0] * (y - box[1]) + (x - box[0])
                image.pixels[ci] = self.pixels[si]

# Line Drawing................................................................

def sign(x):
    return (x > 0) - (x < 0)

def draw_line_without_clipping(image, a, b, colour):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    adx = abs(dx)
    ady = abs(dy)
    sdx = sign(dx)
    sdy = sign(dy)
    x = adx // 2
    y = ady // 2
    px = a[0] # plot x
    py = a[1] # plot y
    image.put_pixel_rgb((px, py), colour)
    if adx >= ady:
	    for _ in range(adx):
		    y += ady;
		    if y >= adx:
			    y -= adx
			    py += sdy
		    px += sdx
		    image.put_pixel_rgb((px, py), colour)
    else:
	    for _ in range(ady):
		    x += adx
		    if x >= ady:
			    x -= ady
			    px += sdx
		    py += sdy
		    image.put_pixel_rgb((px, py), colour)

def clip_test(q, p, te, tl):
	if p == 0:
		return q < 0, te, tl
	t = q / p
	if p > 0:
		if t > tl:
			return False, te, tl
		if t > te:
			te = t
	else:
		if t < te:
			return False, te, tl
		if t < tl:
			tl = t
	return True, te, tl

def clip_line(a, b, box):
    """Clip the line segment (a, b) to the rectangle.
    Uses the Liang–Barsky line clipping algorithm.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    is_point = dx == 0 and dy == 0
    if is_point and not point_in_box(a, box):
	    return False, a, b
    te = 0.0 # entering
    tl = 1.0 # leaving
    q = [
        box[0] - a[0],
        a[0] - (box[2] - 1),
        box[1] - a[1],
        a[1] - (box[3] - 1)]
    p = [dx, -dx, dy, -dy]
    for i in range(4):
        inside, te, tl = clip_test(q[i], p[i], te, tl)
        if not inside:
            return False, a, b
    if tl < 1.0:
        b = (int(a[0] + tl * dx), int(a[1] + tl * dy))
    if te > 0.0:
        a = (int(a[0] + te * dx), int(a[1] + te * dy))
    return True, a, b

def draw_line(image, start, end, colour):
    bounds = image.get_bounds()
    clipped, start, end = clip_line(start, end, bounds)
    if clipped:
        draw_line_without_clipping(image, start, end, colour)

# Flood Fill..................................................................

def try_to_add_vertical(queue, point, offset, image, target_colour, box):
    point = (point[0], point[1] + offset)
    in_bounds = box[1] <= point[1] <= box[3]
    colour_matches = image.get_pixel_rgb(point) == target_colour
    if (in_bounds and colour_matches):
        queue.append(point)

def scan_horizontal(image, point, offset, target_colour, box):
    prior = point
    while True:
        next = (prior[0] + offset, prior[1])
        in_bounds = box[0] <= next[0] <= box[2]
        colour_matches = image.get_pixel_rgb(next) == target_colour
        if not (in_bounds and colour_matches):
            return prior
        prior = next

def flood_fill(image, point, colour):
    target_colour = image.get_pixel_rgb(point)
    if target_colour == colour:
        return
    bounds = image.get_bounds()
    queue = deque([point])
    while len(queue) != 0:
        next = queue.popleft()
        west = scan_horizontal(image, next, -1, target_colour, bounds)
        east = scan_horizontal(image, next,  1, target_colour, bounds)
        line_length = east[0] - west[0] + 1
        for i in range(line_length):
            p = (west[0] + i, west[1])
            image.put_pixel_rgb(p, colour)
            try_to_add_vertical(queue, p, -1, image, target_colour, bounds)
            try_to_add_vertical(queue, p,  1, image, target_colour, bounds)

# Centered Trochoid Drawing...................................................

TAU = 2 * math.pi

def gcd(m, n):
    """Greatest Common Divisor"""
    assert not (m == 0 and n == 0)
    while n:
        m, n = n, m % n
    return m

def draw_hypotrochoid(image, center, R, r, d, colour):
    """
    A curve traced by a point attached to one circle which rolls around the
    inside of another circle.
    Args:
        image = image which will be drawn onto
        center = center of the exterior circle
        R = radius of the larger exterior circle
        r = radius of the smaller interior circle
        d = distance from the drawing point to the center of the interior
            circle
        colour = colour of the line being drawn
    Special Cases:
        R = 2r produces an ellipse.
        d = r proudces a hypocycloid.
    """
    assert R > r
    dr = R - r
    q = dr / r
    center = Vector2.floor(center)
    prior = (int(dr + d), 0)
    prior = Vector2.add(prior, center)
    spr = 20 # segments per revolution
    revolutions = r // gcd(R, r)
    segments = spr * revolutions
    for i in range(segments):
        theta = TAU * i / (spr - 1)
        qt = q * theta
        next = (
            int(dr * math.cos(theta) + d * math.cos(qt)),
            int(dr * math.sin(theta) - d * math.sin(qt)))
        next = Vector2.add(next, center)
        draw_line(image, prior, next, colour)
        prior = next

def draw_epitrochoid(image, center, R, r, d, colour):
    """
    A curve traced by a point attached to one circle which rolls around the
    outside of another circle.
    Args:
        image = image which will be drawn onto
        center = center of the interior circle
        R = radius of the larger interior circle
        r = radius of the smaller exterior circle
        d = distance from the drawing point to the center of the exterior
            circle
        colour = colour of the line being drawn
    Special cases:
        R = r produces a limaçon.
        d = r produces an epicycloid.
    """
    rR = r + R
    q = rR / r
    center = Vector2.floor(center)
    prior = Vector2.add(center, (int(rR + d), 0))
    spr = 20 # segments per revolution
    revolutions = r // gcd(R, r)
    segments = spr * revolutions
    for i in range(segments):
        theta = TAU * i / (spr - 1)
        qt = q * theta
        next = (
            int(rR * math.cos(theta) - d * math.cos(qt)),
            int(rR * math.sin(theta) - d * math.sin(qt)))
        next = Vector2.add(next, center)
        draw_line(image, prior, next, colour)
        prior = next

# Metaball Functions..........................................................

class Metaball:
    def __init__(self, bounds, radius_range, falloff_image):
        radius = random.uniform(radius_range[0], radius_range[1])
        left   = bounds[0] + radius
        top    = bounds[1] + radius
        right  = bounds[2] - radius
        bottom = bounds[3] - radius
        x = random.uniform(left, right)
        y = random.uniform(top, bottom)
        self.center = (x, y)
        self.radius = radius
        self.falloff_image = falloff_image

"""
@Unused

def draw_transparent_metaballs(image, bounds, metaballs):
    for y in range(bounds[1], bounds[3]):
        for x in range(bounds[0], bounds[2]):
            pixel_position = (x, y)
            l = 0
            for metaball in metaballs:
                uv = Vector2.subtract(pixel_position, metaball.center)
                extents = Vector2.divide(metaball.falloff_image.size, 2)
                uv = Vector2.scale(extents[0] / metaball.radius, uv)
                if (abs(uv[0]) <= extents[0] and abs(uv[1]) <= extents[1]):
                    uv = Vector2.add(uv, extents)
                    l += metaball.falloff_image.get_pixel_r(uv)
            l = min(l, 255)
            d_rgb = image.get_pixel_rgb(pixel_position)
            s_rgb = (l, l, l)
            colour = premultiplied_alpha_blend(s_rgb, d_rgb, byte_to_unorm(l))
            image.put_pixel_rgb(pixel_position, colour)
"""

def draw_flat_metaballs(image, bounds, metaballs, colour, threshold):
    for y in range(bounds[1], bounds[3]):
        for x in range(bounds[0], bounds[2]):
            pixel_position = (x, y)
            l = 0
            for metaball in metaballs:
                uv = Vector2.subtract(pixel_position, metaball.center)
                extents = Vector2.divide(metaball.falloff_image.size, 2)
                if (abs(uv[0]) <= extents[0] and abs(uv[1]) <= extents[1]):
                    uv = Vector2.add(uv, extents)
                    uv = Vector2.floor(uv)
                    l += metaball.falloff_image.get_pixel_r(uv)
            l = min(l, 255)
            if l > threshold:
                il = byte_to_unorm(l)
                rgba = (
                    int(il * colour[0]),
                    int(il * colour[1]),
                    int(il * colour[2]),
                    int(l))
                image.put_pixel_rgba(pixel_position, rgba)

def draw_step_metaballs(image, bounds, metaballs, palette, threshold):
    steps = len(palette) - 1
    for y in range(bounds[1], bounds[3]):
        for x in range(bounds[0], bounds[2]):
            pixel_position = (x, y)
            l = 0
            for metaball in metaballs:
                uv = Vector2.subtract(pixel_position, metaball.center)
                extents = Vector2.divide(metaball.falloff_image.size, 2)
                if (abs(uv[0]) <= extents[0] and abs(uv[1]) <= extents[1]):
                    uv = Vector2.add(uv, extents)
                    uv = Vector2.floor(uv)
                    l += metaball.falloff_image.get_pixel_r(uv)
            l = min(l, 255)
            if l > threshold:
                index = int(((l - threshold) / (255 - threshold)) * steps)
                colour = palette[index]
                l /= 255
                rgba = (
                    int(l * colour[0]),
                    int(l * colour[1]),
                    int(l * colour[2]),
                    255)
                image.put_pixel_rgba(pixel_position, rgba)

def make_metaballs(bounds):
    side = 16
    falloff_images = [None] * 3
    for i in range(3):
        falloff_images[i] = Canvas(mode="R", size=(side, side))
    draw_radial_falloff(falloff_images[0])
    draw_polygonal_falloff(falloff_images[1], 3)
    draw_polygonal_falloff(falloff_images[2], 5)
    num_metaballs = 8
    radius_range = (3, 10)
    k = len(falloff_images)
    metaballs = [None] * num_metaballs
    for i in range(num_metaballs):
        metaballs[i] = Metaball(bounds, radius_range, falloff_images[i % k])
    return metaballs

# Random Wave Expression Trees and Scalar/Vector Field Drawing................

# These waves have a period of 1!
def square_wave(x):
    return 4 * math.floor(x) - 2 * math.floor(2 * x) + 1

def triangle_wave(x):
    return abs(4 * (x - math.floor(x)) - 2) - 1

def sawtooth_wave(x):
    return 2 * (x - math.floor(0.5 + x))

def build_expression_tree(probability):
    if random.random() < probability:
        operator_classes = [
            SineOp,
            TriangleOp,
            SawtoothOp,
            MultiplyOp,
            AverageOp]
        return random.choice(operator_classes)(probability)
    else:
        return random.choice([XLeaf, YLeaf])()

class XLeaf:
    def evaluate(self, x, y):
        return x

class YLeaf:
    def evaluate(self, x, y):
        return y

class SineOp:
    def __init__(self, probability):
        self.expression = build_expression_tree(probability ** 2)

    def evaluate(self, x, y):
        return math.sin(TAU * self.expression.evaluate(x, y))

class TriangleOp:
    def __init__(self, probability):
        self.expression = build_expression_tree(probability ** 2)

    def evaluate(self, x, y):
        return triangle_wave(self.expression.evaluate(x, y))

class SawtoothOp:
    def __init__(self, probability):
        self.expression = build_expression_tree(probability ** 2)

    def evaluate(self, x, y):
        return sawtooth_wave(self.expression.evaluate(x, y))

class MultiplyOp:
    def __init__(self, probability):
        self.left = build_expression_tree(probability ** 2)
        self.right = build_expression_tree(probability ** 2)

    def evaluate(self, x, y):
        return self.left.evaluate(x, y) * self.right.evaluate(x, y)

class AverageOp:
    def __init__(self, probability):
        self.left = build_expression_tree(probability ** 2)
        self.right = build_expression_tree(probability ** 2)

    def evaluate(self, x, y):
        return (self.left.evaluate(x, y) + self.right.evaluate(x, y)) / 2

def generate_scalar_field(size):
    field = Canvas(mode="R", size=size)
    wave_expression = build_expression_tree(0.99)
    for y in range(field.size[1]):
        for x in range(field.size[0]):
            u = x / field.size[0]
            v = y / field.size[1]
            t = wave_expression.evaluate(u, v)
            field.put_pixel_r((x, y), snorm_to_byte(t))
    return field

def generate_vector_field(size):
    field = Canvas(mode="RG", size=size)
    limit = 10
    probability = 0.8
    x_expression = build_expression_tree(probability)
    y_expression = build_expression_tree(probability)
    for y in range(field.size[1]):
        for x in range(field.size[0]):
            u = x / field.size[0] - 0.5
            v = y / field.size[1] - 0.5
            d = (
                x_expression.evaluate(u, v),
                y_expression.evaluate(u, v))
            s = Vector2.length(d)
            if s != 0:
                if s > limit:
                    d = Vector2.scale(limit / s, d)
                    d = Vector2.divide(d, limit)
                else:
                    d = Vector2.divide(d, s)
            d = (
                snorm_to_byte(d[0]),
                snorm_to_byte(d[1]))
            field.put_pixel_rg((x, y), d)
    return field

# Brush Generation............................................................

def generate_diamond_square(image, scale):
    assert image.size[0] == image.size[1], "Only square images work here."
    side = image.size[0]
    step = side
    while step >= 1:
        # squares
        for i in range(step, side, step):
            for j in range(step, side, step):
                a = image.get_pixel_f((i - step, j - step))
                b = image.get_pixel_f((i       , j - step))
                c = image.get_pixel_f((i - step, j       ))
                d = image.get_pixel_f((i       , j       ))
                e = (a + b + c + d) / 4.0
                e += random.uniform(-scale, scale)
                image.put_pixel_f((i - step // 2, j - step // 2), e)
        # diamonds
        for i in range(2 * step, side, step):
            for j in range(2 * step, side, step):
                a = image.get_pixel_f((i -     step     , j -     step     ))
                b = image.get_pixel_f((i                , j -     step     ))
                c = image.get_pixel_f((i -     step     , j                ))
                d = image.get_pixel_f((i                , j                ))
                e = image.get_pixel_f((i -     step // 2, j -     step // 2))
                f = image.get_pixel_f((i - 3 * step // 2, j -     step // 2))
                g = image.get_pixel_f((i -     step // 2, j - 3 * step // 2))
                h = (a + c + e + f) / 4.0
                k = (a + b + e + g) / 4.0
                h += random.uniform(-scale, scale)
                k += random.uniform(-scale, scale)
                image.put_pixel_f((i - step     , j - step // 2), h)
                image.put_pixel_f((i - step // 2, j - step     ), k)
        scale /= 2.0
        step //= 2

def get_diamond_square_field(side):
    diamond = Canvas(mode="F", size=(side, side))
    generate_diamond_square(diamond, scale=255)
    return diamond.convert(mode="R")

def map_r_to_rgb(r_image, rgb_image, palette):
    steps = len(palette) - 1
    for y in range(r_image.size[1]):
        for x in range(r_image.size[0]):
            r = byte_to_unorm(r_image.get_pixel_r((x, y)))
            q = steps * r
            i = int(q)
            rgb = rgb_lerp(palette[i], palette[i + 1], q - i)
            rgb_image.put_pixel_rgb((x, y), rgb)

def make_circle_brush(radius):
    side = 2 * radius + 1
    diamond = get_diamond_square_field(side)
    paint = Canvas(mode="RGB", size=diamond.size)
    palette = generate_random_flat_palette(5)
    map_r_to_rgb(diamond, paint, palette)
    alpha = Canvas(mode="R", size=paint.size)
    draw_circle_falloff(alpha, 0.5)
    paint.put_and_premultiply_alpha(alpha)
    return paint

def draw_polygonal_falloff(image, n, margin=(1 / 3)):
    # The polar equation of a regular polygon with n sides and radius 1 is:
    # r(θ) = cos(π/n) / cos(θ % (2*π/n) - π/n)
    pn = math.pi / n
    tn = 2 * pn
    cpn = math.cos(pn)
    for v in range(image.size[1]):
        for u in range(image.size[0]):
            x = 2 * (u + 0.5) / image.size[0] - 1
            y = 2 * (v + 0.5) / image.size[1] - 1
            theta = math.atan2(y, x)
            q = math.cos(theta % tn - pn)
            r = cpn / q
            h = r - math.sqrt((x ** 2) + (y ** 2))
            distance = h * q
            if distance < 0:
                falloff = 0
            elif 0 <= distance < margin:
                falloff = distance / margin
            else:
                falloff = 1
            image.put_pixel_r((u, v), unorm_to_byte(falloff))

def draw_radial_falloff(image):
    side = image.size[0]
    for y in range(side):
        for x in range(side):
            pixel_position = (x, y)
            uv = (x + 0.5, y + 0.5)
            uv = Vector2.divide(uv, side / 2)
            radius = Vector2.distance(uv, (1, 1))
            if 0 <= radius <= (1 / 3):
                falloff = 1 - 3 * radius ** 2
            elif (1 / 3) < radius <= 1:
                falloff = (3 / 2) * (1 - radius) ** 2
            else:
                falloff = 0
            image.put_pixel_r(pixel_position, unorm_to_byte(falloff))

def get_largest_centered_square(bounds):
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if width < height:
        return (
            int(bounds[0]),
            int(bounds[1] + height / 2 - width / 2),
            int(bounds[0] + width),
            int(bounds[1] + width))
    elif height > width:
        return (
            int(bounds[0] + width / 2 - height / 2),
            int(bounds[1]),
            int(bounds[0] + height),
            int(bounds[1] + height))
    else:
        return bounds

def draw_circle_falloff(image, radius):
    box = get_largest_centered_square(image.get_bounds())
    side = box[2] - box[0]
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            pixel_position = (x, y)
            uv = (x + 0.5, y + 0.5)
            uv = Vector2.divide(uv, side / 2)
            d = Vector2.distance(uv, (1, 1))
            if 0 <= d <= radius:
                falloff = 1
            elif radius < d <= 1:
                falloff = 1 - (d - radius) / (1 - radius)
            else:
                falloff = 0
            image.put_pixel_r(pixel_position, unorm_to_byte(falloff))

def multiply_scalar_fields(canvas, image, top_left):
    for y in range(image.size[1]):
        for x in  range(image.size[0]):
            uv = Vector2.add(top_left, (x, y))
            p0 = canvas.get_pixel_r(uv)
            p1 = image.get_pixel_r((x, y))
            result = (p0 * p1) // 255
            canvas.put_pixel_r(uv, result)

def blend_scalar_fields(canvas, image, top_left, alpha):
    for y in range(image.size[1]):
        for x in  range(image.size[0]):
            uv = Vector2.add(top_left, (x, y))
            p0 = canvas.get_pixel_r(uv)
            p1 = image.get_pixel_r((x, y))
            result = lerp(p0, p1, alpha)
            canvas.put_pixel_r(uv, result)

def generate_brush_shape(size):
    kind = random.randint(0, 1)
    shape = Canvas(mode="R", size=size)
    if kind == 0:
        draw_radial_falloff(shape)
    elif kind == 1:
        sides = random.randint(3, 8)
        draw_polygonal_falloff(shape, sides)

# Brushes!....................................................................

class SmudgeBrush:
    def __init__(self, radius):
        side = 2 * radius + 1
        self.paint = Canvas(mode="RGB", size=(side, side))
        self.alpha = Canvas(mode="R", size=self.paint.size)
        draw_radial_falloff(self.alpha)
        self.prior_point = (0.0, 0.0)
        self.radius = radius

    def begin_smudge(self, image, point):
        bounds = get_circle_bounds(point, self.radius)
        image.copy(self.paint, bounds)
        self.prior_point = point

    def smudge_at_point(self, image, point, strength):
        radius = self.radius
        bounds = get_circle_bounds(point, radius)
        bounds = clip_box(bounds, image.get_bounds())
        for y in range(bounds[1], bounds[3]):
            for x in range(bounds[0], bounds[2]):
                s = (
                    int(x - point[0] + radius),
                    int(y - point[1] + radius))
                s_rgb = self.paint.get_pixel_rgb(s)
                alpha = byte_to_unorm(self.alpha.get_pixel_r(s))
                d_rgb = image.get_pixel_rgb((x, y))
                value = rgb_lerp(d_rgb, s_rgb, strength * alpha)
                self.paint.put_pixel_rgb(s, value)
                image.put_pixel_rgb((x, y), value)

    def stroke_smudge(self, image, end_point, strength):
        translation = Vector2.subtract(end_point, self.prior_point)
        d = Vector2.length(translation)
        if d == 0.0:
            return # the brush hasn't moved
        v = Vector2.divide(translation, d)
        steps = int(d)
        for i in range(steps):
            step = Vector2.multiply(i, v)
            next_point = Vector2.add(self.prior_point, step)
            self.smudge_at_point(image, next_point, strength)
        self.smudge_at_point(image, end_point, strength)
        self.prior_point = end_point

class PaintBrush:
    def __init__(self, paint):
        self.paint = paint

    def draw_at_point(self, image, center, angle, opacity=1.0, scale=1.0):
        # To reduce the number of image pixels that need to be iterated over
        # to draw the brush, the bounds of the transformed brush can be used
        # as the region of possible pixels under the brush.
        se = abs(math.sin(angle) / 2)
        ce = abs(math.cos(angle) / 2)
        side_x = self.paint.size[0]
        side_y = self.paint.size[1]
        width  = side_x * scale
        height = side_y * scale
        rotated_extents = (
            height * se + width * ce,
            height * ce + width * se)
        bounds = (
            int(center[0] - rotated_extents[0]),
            int(center[1] - rotated_extents[1]),
            int(center[0] + rotated_extents[0] + 2),
            int(center[1] + rotated_extents[1] + 2))
        bounds = clip_box(bounds, image.get_bounds())
        # These are for stepping the texture coordinate (u, v) along the rows
        # and columns of the source paint image.
        dcol = (math.sin(-angle) / scale, math.cos(-angle) / scale)
        drow = Vector2.perp(dcol)
        # This is the position of the end of the current row.
        row_start = (
            side_x / 2 - (center[0] * dcol[1] + center[1] * dcol[0]),
            side_y / 2 - (center[0] * drow[1] + center[1] * drow[0]))
        row_start = Vector2.add(row_start, Vector2.scale(bounds[1], dcol))
        paint_bounds = self.paint.get_bounds()
        for y in range(bounds[1], bounds[3]):
            uv = Vector2.add(row_start, Vector2.scale(bounds[0], drow))
            for x in range(bounds[0], bounds[2]):
                p = Vector2.floor(uv)
                if point_in_box(p, paint_bounds):
                    s_rgba = self.paint.get_pixel_rgba(p) # source
                    q = (x, y)
                    d_rgb = image.get_pixel_rgb(q) # destination
                    s_a = opacity * byte_to_unorm(s_rgba[3])
                    pixel = premultiplied_alpha_blend(s_rgba, d_rgb, s_a)
                    image.put_pixel_rgb(q, pixel)
                uv = Vector2.add(uv, drow)
            row_start = Vector2.add(row_start, dcol)

    def draw_stroke(self, image, start, end, spacing):
        length = Vector2.length(Vector2.subtract(end, start))
        for step in frange(0, length, spacing):
            point = Vector2.lerp(start, end, step / length)
            angle = random.uniform(-math.pi, math.pi)
            self.draw_at_point(image, point, angle)
        angle = random.uniform(-math.pi, math.pi)
        self.draw_at_point(image, end, angle)

class DistortionBrush:
    def __init__(self, side):
        self.distortion = generate_vector_field(size=(side, side))

    def draw_at_point(self, image, point, strength):
        margin = int(math.ceil(strength))
        side = self.distortion.size[0]
        extent = side / 2 + margin
        copy_bounds = (
            int(point[0] - extent),
            int(point[1] - extent),
            int(point[0] + extent),
            int(point[1] + extent))
        image_bounds = image.get_bounds()
        copy = image.crop(copy_bounds)
        uv_bounds = (
            int(point[0] - side / 2),
            int(point[1] - side / 2),
            int(point[0] + side / 2),
            int(point[1] + side / 2))
        uv_bounds = clip_box(uv_bounds, image_bounds)
        left = uv_bounds[0]
        top = uv_bounds[1]
        uv_bounds = (0, 0, uv_bounds[2] - left, uv_bounds[3] - top)
        copy_bounds = copy.get_bounds()
        for v in range(uv_bounds[1], uv_bounds[3]):
            for u in range(uv_bounds[0], uv_bounds[2]):
                st = self.distortion.get_pixel_rg((u, v))
                st = (
                    strength * byte_to_snorm(st[0]),
                    strength * byte_to_snorm(st[1]))
                sou = Vector2.add((u + margin, v + margin), st)
                sou = Vector2.floor(sou)
                sou = clip_point(sou, copy_bounds)
                des = (left + u, top + v)
                image.put_pixel_rgb(des, copy.get_pixel_rgb(sou))

    def draw_stroke(self, image, start, end, spacing):
        length = Vector2.length(Vector2.subtract(end, start))
        strength = 40
        for step in frange(0, length, spacing):
            point = Vector2.lerp(start, end, step / length)
            self.draw_at_point(image, point, strength)
        self.draw_at_point(image, end, strength)

# Test Functions?.............................................................

def debug_draw_image(canvas, image, top_left):
    image = image.convert("RGB")
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            p = image.get_pixel_rgb((x, y))
            canvas.put_pixel_rgb(Vector2.add(top_left, (x, y)), p)

def debug_draw_palette(image, palette, top_left):
    width = 20
    height = 20
    for i in range(len(palette)):
        for y in range(height):
            for x in range(width):
                point = Vector2.add(top_left, (x, height * i + y))
                image.put_pixel_rgb(point, palette[i])

def test_smudge_brush(image, point_collection):
    smudger = SmudgeBrush(4)
    strength = 0.5
    start_point = random.choice(point_collection)
    smudger.begin_smudge(image, start_point)
    for i in range(5):
        point = random.choice(point_collection)
        smudger.stroke_smudge(image, point, strength)

def test_hypotrochoid(image, image_palette, point_collection):
    kind = random.randint(0, 2)
    if kind == 0: # general hypotrochoid
        r = random.randint(20, 90)
        R = random.randint(r + 1, 92)
        d = random.randint(40, 90)
    elif kind == 1: # ellipse
        r = random.randint(20, 45)
        R = 2 * r
        d = random.randint(40, 90)
    elif kind == 2: # hypocycloid
        r = random.randint(20, 90)
        R = random.randint(r + 1, 92)
        d = r
    curves = 4
    """
    start_hue = random.uniform(0, 3)
    rotations = random.choice([-1, 1]) * random.uniform(0, 2)
    colours, low, high = cube_helix(
        levels=curves,
        start_hue=start_hue,
        rotations=rotations)
    """
    colours = image_palette
    separation = random.randint(1, 5)
    center = random.choice(point_collection)
    for i in range(curves):
        colour = colours[i]
        d += separation
        draw_hypotrochoid(image, center=center, R=R, r=r, d=d, colour=colour)

def test_epitrochoid(image, point_collection):
    r = random.randint(20, 50)
    R = random.randint(20, 50)
    d = random.randint(10, 80)
    # Generate the palette.
    curves = 5
    if random.randint(0, 1):
        colours = generate_roughly_increasing_palette(curves)
    else:
        start_hue = random.uniform(0, 3)
        rotations = random.choice([-1, 1]) * random.uniform(0, 2)
        colours, low, high = cube_helix(
            levels=curves,
            start_hue=start_hue,
            rotations=rotations)        
    # Draw all the curves.
    separation = random.randint(1, 4)
    center = random.choice(point_collection)
    for i in range(curves):
        colour = colours[i]
        d += separation
        draw_epitrochoid(image, center=center, R=R, r=r, d=d, colour=colour)

def test_metaballs(image, point_collection):
    paint = Canvas(mode="RGBA", size=(40, 40))
    bounds = paint.get_bounds()
    metaballs = make_metaballs(bounds)
    steps = random.randint(1, 5)
    threshold = random.randint(50, 200)
    palette = generate_random_flat_palette(steps)
    draw_step_metaballs(paint, bounds, metaballs, palette, threshold)
    brush = PaintBrush(paint)
    start = random.choice(point_collection)
    end = random.choice(point_collection)
    brush.draw_stroke(image, start, end, 20)

def test_waves_expresser(image):
    wavery = generate_scalar_field(size=(64, 64))
    debug_draw_image(image, wavery, (0, 0))

# Spline Functions............................................................

def segment_catmull_rom(p0, p1, p2, p3, segments):
    """Samples a chordal Catmull-Rom curve 1+segments times from p1 to p2."""
    assert p0 != p1 and p1 != p2 and p2 != p3
    points = []
    def tj(ti, pi, pj, alpha):
        xi, yi = pi
        xj, yj = pj
        return math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2) ** alpha + ti
    alpha = 1 # 0.5 makes it centripedal, 1 is chordal
    t0 = 0.0
    t1 = tj(t0, p0, p1, alpha)
    t2 = tj(t1, p1, p2, alpha)
    t3 = tj(t2, p2, p3, alpha)
    def blend(pi, pj, ti, ta, tb):
        return Vector2.add(
            Vector2.scale((ta - ti) / (ta - tb), pi),
            Vector2.scale((ti - tb) / (ta - tb), pj))
    for t in frange(t1, t2, (t2 - t1) / segments):
        a0 = blend(p0, p1, t, t1, t0)
        a1 = blend(p1, p2, t, t2, t1)
        a2 = blend(p2, p3, t, t3, t2)
        b0 = blend(a0, a1, t, t2, t0)
        b1 = blend(a1, a2, t, t3, t1)
        c  = blend(b0, b1, t, t2, t1)
        points.append(c)
    return points

def catmull_rom_spline(p):
    """Args: p - list of points"""
    start = Vector2.add(p[0], Vector2.subtract(p[1], p[0]))
    p.append(start)
    end = Vector2.add(p[-1], Vector2.subtract(p[-2], p[-1]))
    p.insert(0, end)
    segments = 10
    result = []
    for i in range(len(p) - 3):
        curve = segment_catmull_rom(
            p[i], p[i + 1], p[i + 2], p[i + 3], segments)
        result.extend(curve)
    return result

def draw_random_spline(image, point_collection):
    random_points = [random.choice(point_collection) for _ in range(6)]
    random_points = list(set(random_points)) # remove duplicates
    spline = catmull_rom_spline(random_points)
    for i in range(len(spline)):
        spline[i] = Vector2.floor(spline[i])
    palette = generate_random_flat_palette(1)
    for i in range(len(spline) - 1):
        draw_line(image, spline[i], spline[i + 1], palette[0])

def draw_wobbly_stroke(image, brush, start, end, t, spacing):
    length = Vector2.length(Vector2.subtract(end, start))
    dt = random.uniform(0.0001, 0.2)
    for step in frange(0, length, spacing):
        point = Vector2.lerp(start, end, step / length)
        angle = random.uniform(-math.pi, math.pi)
        s = triangle_wave(t) / 2 + 1
        t += dt
        brush.draw_at_point(image, point, angle, scale=s)
    angle = random.uniform(-math.pi, math.pi)
    s = triangle_wave(t) / 2 + 1
    t += dt
    brush.draw_at_point(image, end, angle, scale=s)
    return t

def draw_curvy_stroke(image, brush, start, end, spacing):
    length = Vector2.distance(end, start)
    s = clamp(math.sqrt(length) / 5, 0.1, 1.5)
    for step in frange(0, length, spacing):
        point = Vector2.lerp(start, end, step / length)
        angle = random.uniform(-math.pi, math.pi)
        brush.draw_at_point(image, point, angle, scale=s)

def test_splines(image, point_collection):
    # Make a brush.
    paint = Canvas(mode="RGBA", size=(10, 10))
    bounds = paint.get_bounds()
    metaballs = make_metaballs(bounds)
    threshold = random.randint(50, 200)
    palette = generate_random_flat_palette(1)
    draw_flat_metaballs(paint, bounds, metaballs, palette[0], threshold)
    brush = PaintBrush(paint)
    # Make a random spline.
    random_points = [random.choice(point_collection) for _ in range(6)]
    random_points = list(set(random_points)) # remove duplicates
    spline = catmull_rom_spline(random_points)
    for i in range(len(spline)):
        spline[i] = Vector2.floor(spline[i])
    # Draw the spline, but wobbly.
    #
    # In order for the ends of individual strokes to have the same scale where
    # they meet up, the phase (t) from the end of the prior stroke has to be
    # kept to calculate the starting scale of the next stroke.
    t = 0.0
    for i in range(len(spline) - 1):
        t = draw_wobbly_stroke(image, brush, spline[i], spline[i + 1], t, 7)
    """
    for i in range(len(spline) - 1):
        draw_curvy_stroke(image, brush, spline[i], spline[i + 1], 7)
    """

# Main........................................................................

def main(argv):
    """This here is the real deal."""
    program_name = "mastodon.py"
    parser = argparse.ArgumentParser(
        prog=program_name,
        description="This is an art bot that posts to the microblogging "
        "server Mastodon. It was made by Andrew Dawson.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    parser.add_argument(
        "--email",
        nargs=1,
        required=True,
        help="the login email address for the account this bot should use",
        dest="username")
    parser.add_argument(
        "--password",
        nargs=1,
        required=True,
        help="the login password for the account this bot should use")
    arguments = parser.parse_args(argv)
    # Set the verbose printing global function pointer.
    if arguments.verbose:
        def verbose_print(*args, **kwargs):
            print(*args, **kwargs)
    else:
        verbose_print = lambda *a, **k: None
    global _v_print
    _v_print = verbose_print
    # Establish a connection to Mastodon.
    base_url = "https://mastodon.social"
    client_id, client_secret = get_client_credentials(
        base_url, program_name, "mastodon_client_credentials.txt")
    api = MastodonAPI(base_url, client_id, client_secret)
    """
    access_token = api.log_in(arguments.username, arguments.password)
    _v_print("Obtained access token {}.\n".format(access_token))
    """
    # Open the image generated in the previous run of this bot.
    image_name = "image.png"
    try:
        image = Image.open(image_name)
    except IOError as error:
        print(error)
        print("{} was not able to be opened.".format(image_name))
        return 2
    # Get a basic palette of the image.
    levels = 8
    quantized_image = image.quantize(colors=levels)
    padded_palette = quantized_image.getpalette()
    image_palette = pack_tuple3(padded_palette[:3*levels])
    # Generate random points to use for this iteration.
    random_points = 100
    point_collection = [None] * random_points
    for i in range(random_points):
        point_collection[i] = (random.uniform(0, 511), random.uniform(0, 511))
    # Convert the Pillow image type to our Canvas type.
    pixel_data = image.getdata()
    image_pixels = make_array32(len(pixel_data))
    for i in range(len(pixel_data)):
        image_pixels[i] = pack_rgb(pixel_data[i])
    image = Canvas(mode="RGB", size=image.size, pixels=image_pixels)
    # Smudge that boy up.
    test_smudge_brush(image, point_collection)
    # Test the centered trochoids.
    test_hypotrochoid(image, image_palette, point_collection)
    test_epitrochoid(image, point_collection)
    # Test the paint brush.
    brush = PaintBrush(make_circle_brush(8))
    start = random.choice(point_collection)
    end   = random.choice(point_collection)
    brush.draw_stroke(image, start, end, 6)
    # Test the distortion brush.
    smoosher = DistortionBrush(60)
    start = random.choice(point_collection)
    end   = random.choice(point_collection)
    smoosher.draw_stroke(image, start, end, 30)
    # Test the metaballs.
    test_metaballs(image, point_collection)
    # Test the wave expresser.
    test_waves_expresser(image)
    # Test splines.
    test_splines(image, point_collection)
    # Convert our image type to a Pillow image so it can save it.
    image = Image.frombytes(
        mode="RGBX",
        size=image.size,
        data=image.pixels.tobytes())
    image = image.convert("RGB")
    # Save the image.
    try:
        image.save(image_name, format="PNG")
    except IOError as error:
        print(error)
        print("{} was not be saved.".format(image_name))
        return 2
    # Post the image as a status.
    """
    post_response = api.post_media(image_name)
    media_id = post_response["id"]
    api.post_status("a cloudy toot from a bot", media_ids=[media_id])
    """
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

