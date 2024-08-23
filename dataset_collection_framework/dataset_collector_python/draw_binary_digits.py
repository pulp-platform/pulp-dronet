#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE in the top directory for details.                               #
# You may obtain a copy of the License at                                     #
#                                                                             #
#   http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# File:    draw_binary_digits.py                                              #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniel Rieben    <riebend@student.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from csnake import CodeWriter, Variable, FormattedLiteral


digits = [str(digit) for digit in range(10)]

img_arrays = []
font = ImageFont.truetype(r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
for digit in digits:
    img = Image.new('L', (10, 11), color=(0,))
    draw = ImageDraw.Draw(img)
    draw.text((1, -2), digit, (255,), font=font)
    img_array = np.array(img)
    # plt.imshow(img_array, cmap='gray')
    # plt.show()
    img_arrays.append(img_array)

digit_imgs = np.array(img_arrays)
digits_var = Variable(
    "digits_imgs",
    primitive="uint8_t",
    value=np.array(img_arrays)
)
cw = CodeWriter()
cw.add_variable_initialization(digits_var)
print(cw)