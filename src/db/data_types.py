import sqlalchemy
import sqlalchemy.types as sqltypes

import numpy as np

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
    def process_bind_param(self, values, dialect):
        """
            values is an nparray
        """
        return bytearray(values)

    def process_result_value(self, value, dialect):
        return value
