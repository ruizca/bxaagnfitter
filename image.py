#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:04:16 2021

@author: ruizca
"""
from astropy.io import fits
from astropy.wcs import WCS



class Image:
    def __init__(self, image_file):
        self.wcs, self.data = self._read_image(image_file)


    def _read_image(self, image_file):
        with fits.open(image_file) as hdu:
            wcs = WCS(hdu[0].header)
            data = hdu[0].data

        return wcs, data

    def fill_gaps(self, method="astropy"):
        pass
