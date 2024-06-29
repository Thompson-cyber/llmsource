"""

generated with chatgpt help. 

"create a set of nested pydantic models for this json data, using the key "name" to specify model types"

<json data>
"""

from typing import List, Union, get_type_hints, get_args, Any
from typing import Annotated, Literal
from pydantic import BaseModel, validator, Field
import numpy as np


class UserInput(BaseModel):
    Channel_Request: int = Field(None, alias="Channel Request")
    label: str | None = None
    name: Literal["UserInput"]


class SetVoltage(BaseModel):
    voltage: float
    label: str | None = None
    name: Literal["SetVoltage"]


class Wait(BaseModel):
    time_waited: float
    label: str | None = None
    name: Literal["Wait"]


class Integrate(BaseModel):
    counts: int
    delta_time: float
    coincidences: int
    singles_rate_1: float
    singles_rate_2: float
    coincidence_rate: float
    label: str | None = None
    name: Literal["Integrate"]


class PowerResults(BaseModel):
    expected_amps: float
    voltage_sent: float


class SetPower(BaseModel):
    results: PowerResults
    name: Literal["SetPower"]
    label: str | None = None



class ExtremumResults(BaseModel):
    counts_list: List[int]
    times_list: List[float]
    direction_array: List[Union[float, int]]
    voltage_list: List[float]

class ValueIntegrateExtraData(BaseModel):
    # state: str
    name: str
    label: str
    counts: int
    delta_time: float
    coincidences_hist_1: np.ndarray
    coincidences_hist_2: np.ndarray
    full_coinc_1: np.ndarray
    full_coinc_2: np.ndarray
    singles_hist_1: np.ndarray
    singles_hist_2: np.ndarray
    total_coincidences: int

    # these are heavy arrays, so don't validate every element
    @validator('coincidences_hist_1', 'coincidences_hist_2', 'full_coinc_1', 'full_coinc_2', pre=True)
    def to_numpy_float(cls, value):
        return np.array(value, dtype=float)
    
    @validator('singles_hist_1', 'singles_hist_2', pre=True)
    def to_numpy_int(cls, value):
        return np.array(value, dtype=int)
    
    class Config:
        arbitrary_types_allowed = True


class Extremum(BaseModel):
    results: ExtremumResults
    integration_results: List[ValueIntegrateExtraData]
    label: str | None = None
    name: Literal["Extremum"]

class SimpleSet(BaseModel):
    results: List[ValueIntegrateExtraData]
    label: str | None = None
    name: Literal["SimpleSet"]


class CombineStringStores(BaseModel):
    new_store: str = Field(None, alias="new store")
    label: str | None = None
    name: Literal["CombineStringStores"]


class DerivedVoltages(BaseModel):
    derived_max_voltage: float
    derived_90_voltage: float
    label: str | None = None
    name: Literal["DerivedVoltages"]

Measurement = Annotated[
    Union[
        UserInput,
        SetVoltage,
        Wait,
        Extremum,
        SetPower,
        Integrate,
        CombineStringStores,
        DerivedVoltages,
        SimpleSet
    ],
    Field(discriminator="name"),
]


class DensityMatrixData(BaseModel):
    state: str
    name: str | None = None
    label: str | None = None
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

class ScanStep(BaseModel):
    name: Literal["ScanStep"]
    label: str | None = None
    results: List[Measurement]

class ScanStepTest(BaseModel):
    name: str | None = None
    label: str | None = None
    results: List[dict]
    
class PowerRamp(BaseModel):
    state: str
    name: str | None = None
    label: str | None = None
    results: List[Annotated[Union[ScanStep, SetPower], Field(discriminator="name")]]


def pretty_print_model(model: Any, key="", indent = 0, max_depth = None):
    """pretty print a nested model

    Args:
        model (_type_): object that is recursively parsed
        key (str, optional): to handle indents well, keys are injected to the lower level recursion. Defaults to "".
        indent (int, optional): recursion level used for pretty indented printing. Defaults to 0.
    """
    ind = "  "*indent

    if isinstance(model, str) or isinstance(model, int) or isinstance(model, float):
        if key is not None:
            print(ind, key, ":",end="")
        print(model)
        return
    if isinstance(model, list):
        if key is not None:
            print(ind, key, "<list>")
        if (max_depth is not None) and (max_depth <= indent):
            print(ind,"...")
            return
        for i, item in enumerate(model):
            pretty_print_model(item, key=str(i), indent=indent+1, max_depth=max_depth)
        return
    if isinstance(model, np.ndarray):
        if key is not None:
            print(ind, key, ":",end="")
        print("...data...type:", type(model[0]))
        return
    if model is None:
        if key is not None:
            print(ind, key, ":",end="")
        print("None")
    else:
        if key is not None:
            print(ind, key, "-- ",)
        fields = model.__fields__

        # do the name field first:
        for key, value in fields.items():
            if key == "name":
                # parse_model(getattr(model, key), key=key, indent=indent+2, max_depth=max_depth)
                print(ind, "name: " + '\033[1m' + str(getattr(model, key)) + '\033[0m')
        # print("indent: ", indent)
        if (max_depth is not None) and (max_depth <= indent):
            print(ind,"...")
            return
        for key, value in fields.items():
            if key != "name":
                pretty_print_model(getattr(model, key), key=key, indent=indent+1, max_depth=max_depth)




# def show(self, level):
#     # for key, value in self.__dict__.items():
#     #     item_type = get_type_hints(type(self))[key]
#     #     print(item_type)

#     #     # if 
#     for key in self.__dict__.keys():
#         if type(str) == type(self.__dict__[key]):
#             print("printing string: ", self.__dict__[key])
#         if type(int) == type(self.__dict__[key]):
#             print("printing int: ", self.__dict__[key])
#         if type(float) == type(self.__dict__[key]):
#             print("printing float: ", self.__dict__[key])
#     #         st += item + f": {self.__dict__[item].show(level-1)} \n"
    

#     # if level > 0:
#     #     st = ""
#     #     for item in self.__dict__.keys():
#     #         st += item + f": {self.__dict__[item].show(level-1)} \n"
#     # else:
#     #     return "...\n"
