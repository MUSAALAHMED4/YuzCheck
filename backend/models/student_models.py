from typing import List
from pydantic import BaseModel


class Student(BaseModel):
    name: str


class StudentsResponse(BaseModel):
    students: List[Student]
