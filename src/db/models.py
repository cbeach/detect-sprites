import json

from sqlalchemy import create_engine
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, MetaData, ForeignKey
from sqlalchemy import Boolean, Integer, LargeBinary, String

from .data_types import BoundingBox, Color, Coord, Mask, Shape

Base = declarative_base()

class PatchM(Base):
    __tablename__ = 'patches'

    id = Column(Integer, autoincrement=True, primary_key=True)
    mask = Column(Mask)
    shape = Column(Shape)
    indirect = Column(Boolean)

    def __repr__(self):
        return f"<Patch(id='{self.id}', mask='{self.mask}', shape='{[int(i) for i in self.shape]}', indirect='{self.indirect}')>"

class NodeM(Base):
    __tablename__ = 'nodes'

    id = Column(Integer, primary_key=True)
    patch = Column(None, ForeignKey('patches.id'))
    color = Column(Color)
    bb = Column(BoundingBox)
    patch_graph = Column(None, ForeignKey('patch_graphs.id'))

    def __repr__(self):
       return f"<Node(id='{self.id}', patch='{self.patch}', color='{self.color}', bounding_box={self.bb})>"

class GameM(Base):
    __tablename__ = 'games'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    platform = Column(String)

    def __repr__(self):
        return f'<Game(id={self.id}, name={self.name})>'

class PatchGraphM(Base):
    __tablename__ = 'patch_graphs'
    id = Column(Integer, primary_key=True)
    #tlc = Column(Coord)
    nodes = relationship('NodeM')
    frame_graph = Column(None, ForeignKey('frame_graphs.id'))

class FrameGraphM(Base):
    __tablename__ = 'frame_graphs'

    id = Column(Integer, primary_key=True)
    game = Column(None, ForeignKey('games.id'))
    play_number = Column(Integer)
    frame_number = Column(Integer)
    patch_graphs = relationship('PatchGraphM')

    def __repr__(self):
       return f"<FrameGraph(game='{self.game}', play_number='{self.play_number}', frame_number='{self.frame_number}', patch_graphs={self.patch_graphs})>"

