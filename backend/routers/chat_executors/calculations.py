"""
ARCA Chat Executors - Calculations (Core)

Unit conversions. Domain-specific calculations (Solverr) moved to
domains/{domain}/executors/.
"""

import logging
from functools import lru_cache
from typing import Dict, Any, Tuple

from errors import (
    handle_tool_errors,
    ValidationError,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1000)
def _cached_unit_convert(value: float, from_unit: str, to_unit: str) -> Tuple[bool, Any, str]:
    """Cached conversion calculation. Returns tuple for hashability."""
    return _do_unit_convert(value, from_unit, to_unit)


@handle_tool_errors("unit_convert")
def execute_unit_convert(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert between units with comprehensive unit conversion support.
    Uses LRU cache for repeated conversions (e.g., same unit pairs with different values)."""
    # Use cached version - result is a tuple that we convert back to dict
    result = _cached_unit_convert(value, from_unit, to_unit)
    if result[0]:  # success
        return {"success": True, "result": result[1], "expression": result[2]}
    else:
        raise ValidationError(
            result[1],  # error message
            details=f"Cannot convert {from_unit} to {to_unit}",
            parameter="units",
            received=f"{from_unit} -> {to_unit}",
        )


def _do_unit_convert(value: float, from_unit: str, to_unit: str) -> Tuple[bool, Any, str]:
    """Internal conversion logic. Returns (success, result_or_error, expression_or_none)."""
    conversions = {
        # ===================
        # LENGTH
        # ===================
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
        ("m", "in"): 39.3701,
        ("in", "m"): 0.0254,
        ("km", "mi"): 0.621371,
        ("mi", "km"): 1.60934,
        ("mm", "in"): 0.0393701,
        ("in", "mm"): 25.4,
        ("cm", "in"): 0.393701,
        ("in", "cm"): 2.54,
        ("m", "mm"): 1000,
        ("mm", "m"): 0.001,
        ("m", "cm"): 100,
        ("cm", "m"): 0.01,
        # ===================
        # VOLUME
        # ===================
        ("m3", "l"): 1000,
        ("l", "m3"): 0.001,
        ("m3", "gal"): 264.172,
        ("gal", "m3"): 0.00378541,
        ("l", "gal"): 0.264172,
        ("gal", "l"): 3.78541,
        ("m3", "ft3"): 35.3147,
        ("ft3", "m3"): 0.0283168,
        # ===================
        # AREA
        # ===================
        ("m2", "ft2"): 10.7639,
        ("ft2", "m2"): 0.092903,
        ("m2", "acres"): 0.000247105,
        ("acres", "m2"): 4046.86,
        ("ha", "acres"): 2.47105,
        ("acres", "ha"): 0.404686,
        ("km2", "mi2"): 0.386102,
        ("mi2", "km2"): 2.58999,
        ("ha", "m2"): 10000,
        ("m2", "ha"): 0.0001,
        # ===================
        # PRESSURE / STRESS (basic)
        # ===================
        ("psi", "kpa"): 6.89476,
        ("kpa", "psi"): 0.145038,
        ("bar", "psi"): 14.5038,
        ("psi", "bar"): 0.0689476,
        ("mpa", "psi"): 145.038,
        ("psi", "mpa"): 0.00689476,
        ("bar", "kpa"): 100,
        ("kpa", "bar"): 0.01,
        ("mpa", "kpa"): 1000,
        ("kpa", "mpa"): 0.001,
        ("atm", "kpa"): 101.325,
        ("kpa", "atm"): 0.00986923,
        # ===================
        # PRESSURE / STRESS
        # ===================
        ("kpa", "psf"): 20.8854,
        ("psf", "kpa"): 0.0478803,
        ("kpa", "ksf"): 0.0208854,
        ("ksf", "kpa"): 47.8803,
        ("kpa", "tsf"): 0.01044,  # tons per square foot
        ("tsf", "kpa"): 95.76,
        ("mpa", "ksi"): 0.145038,
        ("ksi", "mpa"): 6.89476,
        ("mpa", "ksf"): 20.8854,
        ("ksf", "mpa"): 0.0478803,
        ("gpa", "ksi"): 145.038,
        ("ksi", "gpa"): 0.00689476,
        # ===================
        # UNIT WEIGHT / DENSITY
        # ===================
        ("kn/m3", "pcf"): 6.36587,
        ("pcf", "kn/m3"): 0.157087,
        ("kg/m3", "pcf"): 0.0624279,
        ("pcf", "kg/m3"): 16.0185,
        ("g/cm3", "pcf"): 62.4279,
        ("pcf", "g/cm3"): 0.0160185,
        ("kn/m3", "kg/m3"): 101.972,
        ("kg/m3", "kn/m3"): 0.00980665,
        ("g/cm3", "kg/m3"): 1000,
        ("kg/m3", "g/cm3"): 0.001,
        ("g/cm3", "kn/m3"): 9.80665,
        ("kn/m3", "g/cm3"): 0.101972,
        # ===================
        # PERMEABILITY / HYDRAULIC CONDUCTIVITY
        # ===================
        ("m/s", "cm/s"): 100,
        ("cm/s", "m/s"): 0.01,
        ("m/s", "ft/day"): 283465,
        ("ft/day", "m/s"): 3.52778e-6,
        ("cm/s", "ft/min"): 1.9685,
        ("ft/min", "cm/s"): 0.508,
        ("m/s", "ft/min"): 196.85,
        ("ft/min", "m/s"): 0.00508,
        ("m/s", "m/day"): 86400,
        ("m/day", "m/s"): 1.1574e-5,
        # ===================
        # FORCE
        # ===================
        ("kn", "kip"): 0.224809,
        ("kip", "kn"): 4.44822,
        ("kn", "lbf"): 224.809,
        ("lbf", "kn"): 0.00444822,
        ("mn", "ton"): 112.404,  # short tons
        ("ton", "mn"): 0.00889644,
        ("kn", "ton"): 0.112404,
        ("ton", "kn"): 8.89644,
        ("n", "lbf"): 0.224809,
        ("lbf", "n"): 4.44822,
        ("kn", "n"): 1000,
        ("n", "kn"): 0.001,
        # ===================
        # MOMENT / TORQUE
        # ===================
        ("kn-m", "kip-ft"): 0.737562,
        ("kip-ft", "kn-m"): 1.35582,
        ("kn-m", "lb-ft"): 737.562,
        ("lb-ft", "kn-m"): 0.00135582,
        ("n-m", "lb-ft"): 0.737562,
        ("lb-ft", "n-m"): 1.35582,
        ("kn-m", "n-m"): 1000,
        ("n-m", "kn-m"): 0.001,
        # ===================
        # MASS
        # ===================
        ("kg", "lb"): 2.20462,
        ("lb", "kg"): 0.453592,
        ("tonne", "ton"): 1.10231,  # metric tonne to short ton
        ("ton", "tonne"): 0.907185,
        ("kg", "ton"): 0.00110231,
        ("ton", "kg"): 907.185,
        ("tonne", "kg"): 1000,
        ("kg", "tonne"): 0.001,
        # ===================
        # THERMAL
        # ===================
        ("w/m-k", "btu/hr-ft-f"): 0.577789,
        ("btu/hr-ft-f", "w/m-k"): 1.73073,
        ("w/m-k", "kcal/m-hr-c"): 0.859845,
        ("kcal/m-hr-c", "w/m-k"): 1.163,
        ("j/kg-k", "btu/lb-f"): 0.000238846,
        ("btu/lb-f", "j/kg-k"): 4186.8,
        ("c-days", "f-days"): 1.8,  # freezing/thawing index
        ("f-days", "c-days"): 0.555556,
        # ===================
        # FLOW RATE
        # ===================
        ("m3/s", "cfs"): 35.3147,
        ("cfs", "m3/s"): 0.0283168,
        ("l/s", "gpm"): 15.8503,
        ("gpm", "l/s"): 0.0630902,
        ("m3/s", "gpm"): 15850.3,
        ("gpm", "m3/s"): 6.30902e-5,
        ("l/s", "cfs"): 0.0353147,
        ("cfs", "l/s"): 28.3168,
        ("m3/s", "l/s"): 1000,
        ("l/s", "m3/s"): 0.001,
        # ===================
        # VELOCITY
        # ===================
        ("m/s", "ft/s"): 3.28084,
        ("ft/s", "m/s"): 0.3048,
        ("km/h", "mph"): 0.621371,
        ("mph", "km/h"): 1.60934,
        ("m/s", "km/h"): 3.6,
        ("km/h", "m/s"): 0.277778,
        ("m/s", "mph"): 2.23694,
        ("mph", "m/s"): 0.44704,
    }

    # Normalize unit strings
    from_unit = from_unit.lower().replace(" ", "").replace("\u00b7", "-").replace("*", "-")
    to_unit = to_unit.lower().replace(" ", "").replace("\u00b7", "-").replace("*", "-")

    # Handle alternate spellings
    unit_aliases = {
        "celsius": "c",
        "fahrenheit": "f",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "foot": "ft",
        "feet": "ft",
        "inch": "in",
        "inches": "in",
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",
        "mile": "mi",
        "miles": "mi",
        "liter": "l",
        "liters": "l",
        "litre": "l",
        "litres": "l",
        "gallon": "gal",
        "gallons": "gal",
        "kilogram": "kg",
        "kilograms": "kg",
        "pound": "lb",
        "pounds": "lb",
        "lbs": "lb",
        "kilopascal": "kpa",
        "kilopascals": "kpa",
        "megapascal": "mpa",
        "megapascals": "mpa",
        "gigapascal": "gpa",
        "gigapascals": "gpa",
        "kilonewton": "kn",
        "kilonewtons": "kn",
        "meganewton": "mn",
        "meganewtons": "mn",
        "newton": "n",
        "newtons": "n",
        "hectare": "ha",
        "hectares": "ha",
        "acre": "acres",
        "cubicfeet": "ft3",
        "cubicfoot": "ft3",
        "cubicmeter": "m3",
        "cubicmeters": "m3",
        "cubicmetre": "m3",
        "cubicmetres": "m3",
        "squarefeet": "ft2",
        "squarefoot": "ft2",
        "sqft": "ft2",
        "sf": "ft2",
        "squaremeter": "m2",
        "squaremeters": "m2",
        "squaremetre": "m2",
        "sqm": "m2",
    }

    from_unit = unit_aliases.get(from_unit, from_unit)
    to_unit = unit_aliases.get(to_unit, to_unit)

    # Temperature conversions (special case - not linear)
    if from_unit == "c" and to_unit == "f":
        result = (value * 9 / 5) + 32
        return (True, round(result, 2), f"{value}\u00b0C = {round(result, 2)}\u00b0F")
    if from_unit == "f" and to_unit == "c":
        result = (value - 32) * 5 / 9
        return (True, round(result, 2), f"{value}\u00b0F = {round(result, 2)}\u00b0C")

    # Look up direct conversion
    key = (from_unit, to_unit)
    if key in conversions:
        result = value * conversions[key]
        # Format output nicely
        if abs(result) >= 1000 or (abs(result) < 0.01 and result != 0):
            result_str = f"{result:.4e}"
        else:
            result_str = f"{round(result, 4)}"
        return (True, round(result, 6), f"{value} {from_unit} = {result_str} {to_unit}")

    # Same unit - no conversion needed
    if from_unit == to_unit:
        return (True, value, f"{value} {from_unit} = {value} {to_unit}")

    return (False, f"Don't know how to convert {from_unit} to {to_unit}", None)


