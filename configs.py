from tests.classes import Thermometer
from hypster import lazy, Options

temperature = Options({"low" : 0.01, "medium" : 0.1, "high" : 1.0}, default="medium")
thermometer = Thermometer(temperature=temperature, location="home")
