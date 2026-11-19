from uuid import *
from pydantic import BaseModel

class MetroCity(BaseModel) :
    city_id: UUID
    lat: float
    lng:    float
    population_weight: float

    def __init__ (self, lat: float, lng: float, population_weight: float) :
        super().__init__(
            city_id= uuid4(), 
            lat=lat, 
            lng=lng, 
            population_weight=population_weight)