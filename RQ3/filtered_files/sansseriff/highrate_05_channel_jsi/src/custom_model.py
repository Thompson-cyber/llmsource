"""

generated with chatgpt with the prompt:

"create a set of nested pydantic models for this json data, using the key "name" to specify model types"

<json data>
"""

from typing import List, Union, Optional
from typing import Annotated, Literal
from pydantic import BaseModel, validator, Field


class UserInput(BaseModel):
    Channel_Request: int = Field(None, alias="Channel Request")
    name: Literal["UserInput"]


class SetVoltage(BaseModel):
    voltage: float
    name: Literal["SetVoltage"]


class Wait(BaseModel):
    time_waited: float
    name: Literal["Wait"]


class Integrate(BaseModel):
    counts: int
    delta_time: float
    coincidences: int
    singles_rate_1: float
    singles_rate_2: float
    coincidence_rate: float
    name: Literal["Integrate"]


class PowerResults(BaseModel):
    expected_amps: float
    voltage_sent: float


class SetPower(BaseModel):
    result: PowerResults
    name: Literal["SetPower"]


class ExtremumResults(BaseModel):
    counts_list: List[int]
    times_list: List[float]
    direction_array: List[Union[float, int]]
    voltage_list: List[float]


class Extremum(BaseModel):
    results: ExtremumResults
    integration_results: List[Integrate]
    name: Literal["Extremum"]


class CombineStringStores(BaseModel):
    new_store: str = Field(None, alias="new store")
    name: Literal["CombineStringStores"]


Measurement = Annotated[
    Union[
        UserInput,
        SetVoltage,
        Wait,
        Extremum,
        SetPower,
        Integrate,
        CombineStringStores,
    ],
    Field(discriminator="name"),
]


class FastMinimum(BaseModel):
    state: str
    name: Optional[str] = None
    results: List[Measurement]

    # Use a validator to check that each result is of the correct type for its name
    # @validator("results")
    # def check_results(cls, results):
    #     for result in results:
    #         if result.name == "UserInput":
    #             assert isinstance(result, UserInput)
    #         elif result.name == "SetVoltage":
    #             assert isinstance(result, SetVoltage)
    #         elif result.name == "Wait":
    #             assert isinstance(result, Wait)
    #         elif result.name == "Integrate":
    #             assert isinstance(result, Integrate)
    #         elif result.name == "SetPower":
    #             assert isinstance(result, SetPower)
    #         elif result.name == "Extremum":
    #             assert isinstance(result, Extremum)
    #     return results
