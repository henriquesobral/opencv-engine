#!/usr/bin/env python
# -*- coding: utf-8 -*-

# thumbor imaging service - opencv engine
# https://github.com/thumbor/opencv-engine

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2014 globo.com timehome@corp.globo.com
from opencv_engine.tiff_support import TiffMixin, TIFF_FORMATS

try:
    import cv
except ImportError:
    import cv2.cv2 as cv

from colour import Color

from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment
<<<<<<< HEAD
from types import MethodType
import cv2
import gdal
import numpy
from osgeo import osr

import thumbor.utils
thumbor.utils.CONTENT_TYPE = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.gif': 'image/gif',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.svg': 'image/svg+xml',
}
thumbor.utils.EXTENSION = {
    'image/jpeg': '.jpg',
    'image/tiff': '.tif',
    'image/gif': '.gif',
    'image/png': '.png',
    'image/webp': '.webp',
    'video/mp4': '.mp4',
    'video/webm': '.webm',
    'image/svg+xml': '.svg',
}
# need to monkey patch the BaseEngine.get_mimetype function to handle tiffs
# has to be patched this way b/c called as both a classmethod and instance method internally in thumbor
old_mime = BaseEngine.get_mimetype


def new_mime(buffer):
    ''' determine the mime type from the raw image data
        Args:
            buffer - raw image data
        Returns:
            mime - mime type of image
    '''
    mime = old_mime(buffer)
    # tif files start with 'II'
    if not mime and buffer.startswith('II'):
        mime = 'image/tiff'
    return mime

BaseEngine.get_mimetype = staticmethod(new_mime)
=======
>>>>>>> 6a06379c82969b0d989b56b6d46d66930aa1f2a3

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.jpg': 'JPEG',
    '.JPG': 'JPEG',
    '.jpeg': 'JPEG',
    '.JPEG': 'JPEG',
    '.gif': 'GIF',
<<<<<<< HEAD
    '.GIF': 'GIF',
    '.png': 'PNG',
    '.PNG': 'PNG',
    '.tiff': 'TIFF',
    '.TIFF': 'TIFF',
    '.tif': 'TIFF',
    '.TIF': 'TIFF',
=======
    '.png': 'PNG'
>>>>>>> 6a06379c82969b0d989b56b6d46d66930aa1f2a3
}

FORMATS.update(TIFF_FORMATS)


class Engine(BaseEngine, TiffMixin):

    @property
    def image_depth(self):
        if self.image is None:
            return 8
        return cv.GetImage(self.image).depth

    @property
    def image_channels(self):
        if self.image is None:
            return 3
        return self.image.channels

    @classmethod
    def parse_hex_color(cls, color):
        try:
            color = Color(color).get_rgb()
            return tuple(c * 255 for c in reversed(color))
        except Exception:
            return None

    def gen_image(self, size, color_value):
        img0 = cv.CreateImage(size, self.image_depth, self.image_channels)
        if color_value == 'transparent':
            color = (255, 255, 255, 255)
        else:
            color = self.parse_hex_color(color_value)
            if not color:
                raise ValueError('Color %s is not valid.' % color_value)
        cv.Set(img0, color)
        return img0

<<<<<<< HEAD
    def read(self, extension=None, quality=None):
        if not extension and FORMATS[self.extension] == 'TIFF':
            # If the image loaded was a tiff, return the buffer created earlier.
            return self.buffer
        else:
            if quality is None:
                quality = self.context.config.QUALITY
            options = None
            self.extension = extension or self.extension

            # Check if we should write a JPEG. If we are allowing defaulting to jpeg
            # and if the alpha channel is all white (opaque).
            channels = None
            if hasattr(self.context, 'request') and getattr(self.context.request, 'default_to_jpeg', True):
                if self.image_channels > 3:
                    # import pdb; pdb.set_trace()
                    if type(self.image) == cv2.cv.iplimage:
                        arr = numpy.asarray(self.image[:,:])
                    else:
                        arr = numpy.asarray(self.image)
                    channels = cv2.split(arr)
                    if numpy.all(channels[3] == 255):
                        self.extension = '.jpg'

            try:
                if FORMATS[self.extension] == 'JPEG':
                    options = [cv.CV_IMWRITE_JPEG_QUALITY, quality]
            except KeyError:
                # default is JPEG so
                options = [cv.CV_IMWRITE_JPEG_QUALITY, quality]

            if FORMATS[self.extension] == 'TIFF':
                channels = channels or cv2.split(numpy.asarray(self.image))
                data = self.write_channels_to_tiff_buffer(channels)
            else:
                data = cv.EncodeImage(self.extension, self.image, options or []).tostring()

            if FORMATS[self.extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
                if hasattr(self, 'exif'):
                    img = JpegFile.fromString(data)
                    img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                    data = img.writeString()

        return data

=======
>>>>>>> 6a06379c82969b0d989b56b6d46d66930aa1f2a3
    def create_image(self, buffer, create_alpha=True):
        self.extension = self.extension or '.tif'
        self.no_data_value = None
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        try:
            if FORMATS[self.extension] == 'GIF':
                raise ValueError("opencv doesn't support gifs")
        except KeyError:
            pass

        if FORMATS[self.extension] == 'TIFF':
            self.buffer = buffer
            img0 = self.read_tiff(buffer, create_alpha)
        else:
            imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
            cv.SetData(imagefiledata, buffer, len(buffer))
            img0 = cv.DecodeImageM(imagefiledata, cv.CV_LOAD_IMAGE_UNCHANGED)

        if FORMATS[self.extension] == 'JPEG':
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass

<<<<<<< HEAD
    def read_tiff(self, buffer, create_alpha=True):
        """ Reads image using GDAL from a buffer, and returns a CV2 image.
        """

        if hasattr(self.context, 'offset') and self.context.offset != 0:
            offset = float(self.context.offset)
        else:
            offset = 0

        mem_map_name = '/vsimem/{}'.format(uuid.uuid4().get_hex())
        gdal_img = None
        try:
            gdal.FileFromMemBuffer(mem_map_name, buffer)
            gdal_img = gdal.Open(mem_map_name)

            channels = [gdal_img.GetRasterBand(i).ReadAsArray() for i in range(1, gdal_img.RasterCount + 1)]

            if len(channels) >= 3:  # opencv is bgr not rgb.
                red_channel = channels[0]
                channels[0] = channels[2]
                channels[2] = red_channel + offset

            if len(channels) < 4 and create_alpha:
                self.no_data_value = gdal_img.GetRasterBand(1).GetNoDataValue()
                channels.append(numpy.float32(gdal_img.GetRasterBand(1).GetMaskBand().ReadAsArray()))
            return cv2.merge(channels)
        finally:
            gdal_img = None
            gdal.Unlink(mem_map_name)  # Cleanup.

    def read_vsimem(self, fn):
        """Read GDAL vsimem files"""
        vsifile = None
        try:
            vsifile = gdal.VSIFOpenL(fn, 'r')
            gdal.VSIFSeekL(vsifile, 0, 2)
            vsileng = gdal.VSIFTellL(vsifile)
            gdal.VSIFSeekL(vsifile, 0, 0)
            return gdal.VSIFReadL(1, vsileng, vsifile)
        finally:
            if vsifile:
                gdal.VSIFCloseL(vsifile)

    def write_channels_to_tiff_buffer(self, channels):
        mem_map_name = '/vsimem/{}.tiff'.format(uuid.uuid4().get_hex())
        driver = gdal.GetDriverByName('GTiff')
        w, h = channels[0].shape
        gdal_img = None
        try:
            if len(channels) == 1:
                # DEM Tiff (32 bit floating point single channel)
                gdal_img = driver.Create(mem_map_name, w, h, len(channels), gdal.GDT_Float32)
                outband = gdal_img.GetRasterBand(1)
                outband.WriteArray(channels[0], 0, 0)
                outband.SetNoDataValue(-32767)
                outband.FlushCache()
                outband = None
                gdal_img.FlushCache()

                self.set_geo_info(gdal_img)
                return self.read_vsimem(mem_map_name)
            elif len(channels) == 4:
                # BGRA 8 bit unsigned int.
                gdal_img = driver.Create(mem_map_name, h, w, len(channels), gdal.GDT_Byte)
                band_order = [2, 1, 0, 3]
                img_bands = [gdal_img.GetRasterBand(i) for i in range(1, 5)]
                for outband, band_i in zip(img_bands, band_order):
                    outband.WriteArray(channels[band_i], 0, 0)
                    outband.SetNoDataValue(-32767)
                    outband.FlushCache()
                    del outband
                del img_bands

                self.set_geo_info(gdal_img)
                return self.read_vsimem(mem_map_name)
        finally:
            del gdal_img
            gdal.Unlink(mem_map_name)  # Cleanup.

    def set_geo_info(self, gdal_img):
        """ Set the georeferencing information for the given gdal image.
        """
        if hasattr(self.context.request, 'geo_info'):
            geo = self.context.request.geo_info
            gdal_img.SetGeoTransform([geo['upper_left_x'], geo['resx'], 0, geo['upper_left_y'], 0, -geo['resy']])

        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        gdal_img.SetProjection(srs.ExportToWkt())
        gdal_img.FlushCache()
        del srs
=======
        return img0
>>>>>>> 6a06379c82969b0d989b56b6d46d66930aa1f2a3

    @property
    def size(self):
        return cv.GetSize(self.image)

    def normalize(self):
        pass

    def resize(self, width, height):
        dims = (int(round(width, 0)), int(round(height, 0)))
        self.image = cv2.resize(numpy.asarray(self.image), dims, interpolation=cv2.INTER_CUBIC)

    def crop(self, left, top, right, bottom):
        x1, y1 = left, top
        x2, y2 = right, bottom
        self.image = self.image[y1:y2, x1:x2]

    def rotate(self, degrees):
        """ rotates the image by specified number of degrees.
            Uses more effecient flip and transpose for multiples of 90

            Args:
                degrees - degrees to rotate image by (CCW)
        """
        image = numpy.asarray(self.image)
        # number passed to flip corresponds to rotation about: (0) x-axis, (1) y-axis, (-1) both axes
        if degrees == 270:
            transposed = cv2.transpose(image)
            rotated = cv2.flip(transposed, 1)
        elif degrees == 180:
            rotated = cv2.flip(image, -1)
        elif degrees == 90:
            transposed = cv2.transpose(image)
            rotated = cv2.flip(transposed, 0)
        else:
            rotated = self._rotate(image, degrees)

        self.image = cv.fromarray(rotated)

    def _rotate(self, image, degrees):
        """ rotate an image about it's center by an arbitrary number of degrees

            Args:
                image - image to rotate (CvMat array)
                degrees - number of degrees to rotate by (CCW)

            Returns:
                rotated image (numpy array)
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def flip_vertically(self):
        """ flip an image vertically (about x-axis) """
        image = numpy.asarray(self.image)
        self.image = cv.fromarray(cv2.flip(image, 0))

    def flip_horizontally(self):
        """ flip an image horizontally (about y-axis) """
        image = numpy.asarray(self.image)
        self.image = cv.fromarray(cv2.flip(image, 1))

    def read(self, extension=None, quality=None):
        if not extension and FORMATS[self.extension] == 'TIFF':
            # If the image loaded was a tiff, return the buffer created earlier.
            return self.buffer
        else:
            if quality is None:
                quality = self.context.config.QUALITY
            options = None
            self.extension = extension or self.extension

            try:
                if FORMATS[self.extension] == 'JPEG':
                    options = [cv2.IMWRITE_JPEG_QUALITY, quality]
            except KeyError:
                options = [cv2.IMWRITE_JPEG_QUALITY, quality]

            if FORMATS[self.extension] == 'TIFF':
                channels = cv2.split(numpy.asarray(self.image))
                data = self.write_channels_to_tiff_buffer(channels)
            else:
                success, numpy_data = cv2.imencode(self.extension, numpy.asarray(self.image), options or [])
                if success:
                    data = numpy_data.tostring()
                else:
                    raise Exception("Failed to encode image")

            if FORMATS[self.extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
                if hasattr(self, 'exif'):
                    img = JpegFile.fromString(data)
                    img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                    data = img.writeString()

        return data

    def set_image_data(self, data):
        cv.SetData(self.image, data)

    def image_data_as_rgb(self, update_image=True):
        if self.image.channels == 4:
            mode = 'BGRA'
        elif self.image.channels == 3:
            mode = 'BGR'
        else:
            raise NotImplementedError("Only support fetching image data as RGB for 3/4 channel images")
        return mode, self.image.tostring()
