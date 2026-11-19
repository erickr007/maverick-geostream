from uuid import *
from pydantic import BaseModel

class ApplicationSettings(BaseModel):
    """
    Application settings model.
    """
    app_name: str
    version: str
    