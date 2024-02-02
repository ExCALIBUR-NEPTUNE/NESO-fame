from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from enum import Enum
from typing import Generic, Iterator, Optional, TypeVar, overload

from NekPy import LibUtilities

class Geometry:
    def GetCoordim(self) -> int: ...
    def GetGlobalID(self) -> int: ...
    def SetGlobalID(self, val: int, /) -> None: ...
    def Setup(self) -> None: ...
    def FillGeom(self) -> None: ...
    def GenGeomFactors(self) -> None: ...
    def IsValid(self) -> bool: ...
    def ContainsPoint(self, gloCoord: Sequence[float]) -> bool: ...
    def GetVertex(self, i: int, /) -> PointGeom: ...
    def GetEdge(self, i: int, /) -> Geometry1D: ...
    def GetFace(self, i: int, /) -> Geometry2D: ...
    def GetVid(self, i: int, /) -> int: ...
    def GetEid(self, i: int, /) -> int: ...
    def GetFid(self, i: int, /) -> int: ...
    def GetTid(self, i: int, /) -> int: ...
    def GetNumVerts(self) -> int: ...
    def GetNumEdges(self) -> int: ...
    def GetNumFaces(self) -> int: ...
    def GetShapeDim(self) -> int: ...
    def GetShapeType(self) -> LibUtilities.ShapeType: ...

    # Haven't written bindings for StandardRegions yet
    # def GetEorient(self) -> StandardRegions.Orientation: ...
    # def GetForient(self) -> StandardRegions.Orientation: ...
    # def GetXmap(self) -> StdRegions.StdExpansion: ...

    def GetCoeffs(self, i: int, /) -> Sequence[float]: ...

class Geometry1D(Geometry): ...

class Geometry2D(Geometry):
    def GetCurve(self) -> Curve: ...

class Geometry3D(Geometry): ...

class PointGeom(Geometry):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, coordim: int, vid: int, x: float, y: float, z: float, /
    ) -> None: ...
    def GetCoordinates(self) -> tuple[float, float, float]: ...

class SegGeom(Geometry1D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        id: int,
        coordim: int,
        points: list[PointGeom] = ...,
        curve: Optional[Curve] = ...,
    ) -> None: ...
    def GetCurve(self) -> Curve: ...

class TriGeom(Geometry2D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[SegGeom] = ...) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[SegGeom], curve: Curve) -> None: ...

class QuadGeom(Geometry2D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[SegGeom] = ...) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[SegGeom], curve: Curve) -> None: ...

class TetGeom(Geometry3D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[TriGeom] = ...) -> None: ...

class PrismGeom(Geometry3D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[Geometry2D] = ...) -> None: ...

class PyrGeom(Geometry3D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[Geometry2D] = ...) -> None: ...

class HexGeom(Geometry3D):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, id: int, segments: list[QuadGeom] = ...) -> None: ...

class Curve:
    curveID: int
    ptype: LibUtilities.PointsType
    points: list[PointGeom]
    def __init__(self, curveID: int, type: LibUtilities.PointsType, /) -> None: ...

class Composite:
    geometries: MutableSequence[Geometry]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, geometries: list[Geometry], /) -> None: ...

T = TypeVar("T")

class _NekMapItem(Generic[T]):
    def key(self) -> int: ...
    def data(self) -> T: ...

class _NekMap(Generic[T]):
    def __init__(self) -> None: ...
    def __delitem__(self, key: int, /) -> None: ...
    def __getitem__(self, key: int, /) -> T: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: int, value: T) -> None: ...
    def __contains__(self, key: int) -> bool: ...
    def __iter__(self) -> Iterator[_NekMapItem[T]]: ...

PointGeomMap = _NekMap[PointGeom]
SegGeomMap = _NekMap[SegGeom]
TriGeomMap = _NekMap[TriGeom]
QuadGeomMap = _NekMap[QuadGeom]
TetGeomMap = _NekMap[TetGeom]
PrismGeomMap = _NekMap[PrismGeom]
PyrGeomMap = _NekMap[PyrGeom]
HexGeomMap = _NekMap[HexGeom]
CurveMap = _NekMap[Curve]
CompositeMap = _NekMap[Composite]

class MeshGraph:
    @staticmethod
    def Read(session: LibUtilities.SessionReader) -> MeshGraph: ...
    def Write(
        self, outfile: str, defaultExp: bool = ..., metadata: FieldMetaDataMap = ...
    ) -> None: ...
    def GetMeshDimension(self) -> int: ...
    def GetAllPointGeoms(self) -> PointGeomMap: ...
    def GetAllSegGeoms(self) -> SegGeomMap: ...
    def GetAllTriGeoms(self) -> TriGeomMap: ...
    def GetAllQuadGeoms(self) -> QuadGeomMap: ...
    def GetAllTetGeoms(self) -> TetGeomMap: ...
    def GetAllPyrGeoms(self) -> PyrGeomMap: ...
    def GetAllPrismGeoms(self) -> PrismGeomMap: ...
    def GetAllHexGeoms(self) -> HexGeomMap: ...
    def GetCurvedEdges(self) -> CurveMap: ...
    def GetCurvedFaces(self) -> CurveMap: ...
    def GetComposites(self) -> CompositeMap: ...
    def GetDomain(self) -> _NekMap[CompositeMap]: ...
    def GetMovement(self) -> Movement: ...
    def GetNumElements(self) -> int: ...
    def SetExpansionInfosToEvenlySpacedPoints(self, npoints: int, /) -> None: ...
    def SetExpansionInfosToPolyOrder(self, nmodes: int, /) -> None: ...
    def SetExpansionInfoToPointOrder(self, npts: int, /) -> None: ...

class MeshGraphXml(MeshGraph):
    def __init__(self, meshDim: int, spatialDim: int, /) -> None: ...

class MeshGraphXmlCompressed(MeshGraphXml):
    def __init__(self, meshDim: int, spatialDim: int, /) -> None: ...

class MovementType(Enum):
    Fixed = ...
    Rotate = ...
    Translate = ...
    Prescribe = ...

class Zone:
    def GetMovementType(self) -> MovementType: ...
    def GetDomain(self) -> CompositeMap: ...
    def GetID(self) -> int: ...
    def GetDomainID(self) -> int: ...
    def Move(self) -> bool: ...
    def GetElements(self) -> Sequence[Geometry]: ...
    def GetMoved(self) -> bool: ...
    def ClearBoundingBoxes(self) -> None: ...

class ZoneFixed(Zone):
    def __init__(
        self, id: int, domainID: int, domain: CompositeMap, coordDim: int, /
    ) -> None: ...

# TODO: Add stubs for other Zone types

class Interface:
    def __init__(self, indx: int, edge: CompositeMap, /) -> None: ...
    def GetEdge(self, id: int, /) -> Geometry: ...
    def IsEmpty(self) -> bool: ...
    def GetId(self) -> int: ...
    def GetOppInterface(self) -> Interface: ...
    def GetCompositeIDs(self) -> Sequence[int]: ...

class InterfacePair:
    def __init__(
        self, leftInterface: Interface, rightInterface: Interface, /
    ) -> None: ...
    def GetLeftInterface(self) -> Interface: ...
    def GetRightInterface(self) -> Interface: ...

ZoneMap = _NekMap[Zone]

class Movement:
    def __init__(self) -> None: ...
    def GetInterfaces(self) -> dict[tuple[int, str], InterfacePair]: ...
    def GetZones(self) -> ZoneMap: ...
    def PerformMovement(self, timeStep: float, /) -> None: ...
    def AddZone(self, zone: Zone, /) -> None: ...
    def AddInterface(self, name: str, left: Interface, right: Interface, /) -> None: ...

class FieldMetaDataMap: ...
