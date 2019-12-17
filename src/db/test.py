import json
import sys

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, MetaData, ForeignKey
from sqlalchemy import Boolean, Integer, String

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.sql import select
from .models import Base, FrameGraphM, GameM, NodeM, PatchM, PatchGraphM
from .data_types import BoundingBox, Mask, Shape

engine = create_engine('sqlite:///:memory:', echo=True)
#engine = create_engine('sqlite:///sqlite.db', echo=True)
Base.metadata.create_all(engine)

conn = engine.connect()
Session = sessionmaker(bind=engine)
s = Session()

with open('./games.json', 'r') as fp:
    games = json.load(fp)

game_list = []
game_list = sorted([*games['NES'], *games['SNES']])
for i, g in enumerate(game_list):
    s.add(GameM(id=i, name=g, platform=g.split('-')[-1].upper()))

for row in s.query(GameM).all():
    print('query: ', row)

smb = s.query(GameM).filter(GameM.name == 'SuperMarioBros-Nes').first()

patch = PatchM(mask=bytearray((1, 0, 1, 0, 1, )), shape=bytearray((2, 2)), direct=False)
s.add(patch)

for row in s.query(PatchM).all():
    print('query: ', row)

n1 = NodeM(id=1, patch=1, color=(1, 1, 1), bb=((1, 1), (3, 3)))
n2 = NodeM(id=2, patch=2, color=(2, 2, 2), bb=((2, 2), (4, 4)))
n3 = NodeM(id=3, patch=3, color=(3, 3, 3), bb=((3, 3), (3, 3)))
n4 = NodeM(id=4, patch=4, color=(4, 4, 4), bb=((4, 4), (3, 3)))
s.add(n1)
s.add(n2)
s.add(n3)
s.add(n4)
for row in s.query(NodeM).all():
    print('query: ', row)

pg1 = PatchGraphM(nodes=[n1, n2])
pg2 = PatchGraphM(nodes=[n3, n4])
s.add(pg1, pg2)
for row in s.query(PatchGraphM).all():
    print('query: ', row)

s.add(FrameGraphM(game=smb.id, play_number=1, frame_number=1, patch_graphs=[pg1]))
s.add(FrameGraphM(game=smb.id, play_number=1, frame_number=1, patch_graphs=[pg2]))

for row in s.query(FrameGraphM).all():
    print('query: ', row)
