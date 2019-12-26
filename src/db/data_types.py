import sqlalchemy
import sqlalchemy.types as sqltypes

import numpy as np

def byte_to_list(b):
    """
        Convert a number in range(0, 256) (byte) to a list of bool as int [01] corresponding to the binary
        expression of the number.
        Eg:
            b =  78 = 0b01001110 = [False, True, False, False, True, True, True, False]
            b = 170 = 0b10101010 = [True, False, True, False, True, False, True, False]
    """
    if b > 255:
        raise ValueError('byte must be in range(0, 256)')
    return [bool(ord(i) - 48) for i in '{:0>8b}'.format(b)]

class Shape(sqltypes.TypeDecorator):
    impl = sqltypes.LargeBinary
    def process_bind_param(self, values, dialect):
        return bytearray(values)

    def process_result_value(self, value, dialect):
        return list(value)

class BoundingBox(sqltypes.TypeDecorator):
    impl = sqltypes.LargeBinary
    def __init__(self, *args, **kwargs):
        super(BoundingBox, self).__init__(length=4)

    def process_bind_param(self, value, dialect):
        return bytearray((value[0][0], value[0][1], value[1][0], value[1][1]))

    def process_result_value(self, value, dialect):
        return ((value[0], value[1]), (value[2], value[3]))

class Color(sqltypes.TypeDecorator):
    impl = sqltypes.LargeBinary
    def __init__(self, *args, **kwargs):
        super(Color, self).__init__(length=5)

    def process_bind_param(self, values, dialect):
        return bytearray(values)

    def process_result_value(self, value, dialect):
        if len(value) > 3:
            return (value[0], value[1], value[2], value[3])
        else:
            return (value[0], value[1], value[2])

class Coord(sqltypes.TypeDecorator):
    impl = sqltypes.LargeBinary
    def __init__(self, *args, **kwargs):
        super(Coord, self).__init__(length=2)

    def process_bind_param(self, values, dialect):
        return bytearray(values)

    def process_result_value(self, value, dialect):
        return (value[0], value[1])

class Mask(sqltypes.TypeDecorator):
    impl = sqltypes.LargeBinary
    bytes_as_list = [byte_to_list(i) for i in range(256)]
    def process_bind_param(self, values, dialect):
        """
            values is an nparray
        """
        f, t, s, fi = 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff
        x, y = values.shape
        ba = bytearray([(x & f) >> 24, (x & t) >> 16, (x & s) >> 8, (x & fi), (y & f) >> 24, (y & t) >> 16, (y & s) >> 8, (y & fi)])

        counter = b = 0
        for i in values.flatten():
            b = b << 1
            b += i

            if counter == 7:
                ba.append(b)
                counter = 0
                b = 0
                continue
            else:
                counter += 1

        if counter is not 0:
            b = b << (8 - counter)
            ba.append(b)

        return ba

    def process_result_value(self, value, dialect):
        value = bytearray(value)
        print(value)
        x = y = 0
        x |= value.pop(0) << 24
        x |= value.pop(0) << 16
        x |= value.pop(0) << 8
        x |= value.pop(0)

        y |= value.pop(0) << 24
        y |= value.pop(0) << 16
        y |= value.pop(0) << 8
        y |= value.pop(0)

        length = x * y
        if length < 8:
            arr = np.array(Mask.bytes_as_list[value[0]][:length])
        else:
            arr = np.zeros((length), dtype=bool)
            for i, b in enumerate(value[:-1]):
                ind = i * 8
                arr[ind:ind + 8] = Mask.bytes_as_list[b]

            mod = length % 8
            if mod == 0:
                arr[-8:] = Mask.bytes_as_list[value[-1]]
            else:
                arr[- mod:] = Mask.bytes_as_list[value[-1]][:mod]

        arr = arr.reshape((x, y))
        return arr
