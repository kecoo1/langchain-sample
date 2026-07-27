from fastmcp import FastMCP

mcp = FastMCP("Weather")

weather_data = {
    "beijing": {"temperature": 28, "condition": "sunny", "humidity": 45},
    "shanghai": {"temperature": 30, "condition": "cloudy", "humidity": 60},
    "guangzhou": {"temperature": 32, "condition": "rainy", "humidity": 80},
    "shenzhen": {"temperature": 31, "condition": "thunderstorm", "humidity": 85},
    "new york": {"temperature": 22, "condition": "windy", "humidity": 50},
    "london": {"temperature": 18, "condition": "foggy", "humidity": 75},
    "tokyo": {"temperature": 26, "condition": "sunny", "humidity": 55},
}

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for a location.
    
    Args:
        location: The name of the city to get weather for.
    """
    location_lower = location.lower().strip()
    
    if location_lower in weather_data:
        data = weather_data[location_lower]
        return f"Weather in {location}: {data['condition']}, {data['temperature']}°C, humidity {data['humidity']}%"
    else:
        return f"Sorry, weather data not available for {location}. Available cities: {', '.join(weather_data.keys())}"

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for a location.
    
    Args:
        location: The name of the city to get forecast for.
        days: Number of days to forecast (default: 3).
    """
    location_lower = location.lower().strip()
    
    if location_lower in weather_data:
        data = weather_data[location_lower]
        conditions = ["sunny", "cloudy", "rainy", "sunny"]
        return f"Forecast for {location} ({days} days):\n" + "\n".join([
            f"  Day {i+1}: {conditions[i % len(conditions)]}, {data['temperature'] + (i-1)}°C"
            for i in range(days)
        ])
    else:
        return f"Sorry, forecast data not available for {location}."

if __name__ == "__main__":
    mcp.run(transport="stdio")