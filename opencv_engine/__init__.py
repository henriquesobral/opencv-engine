#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from thumbor.engines import BaseEngine

__version__ = '1.0.1'

try:
    from opencv_engine.engine_cv3 import Engine  # NOQA
except ImportError:
    logging.exception('Could not import opencv_engine. Probably due to setup.py installing it.')


def _patch_mime_types():
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


_patch_mime_types()
