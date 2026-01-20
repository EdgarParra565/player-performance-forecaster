import requests

def fetch_odds(api_key: str, sport: str = "basketball_nba"):
    """
    Placeholder function to fetch odds from a sportsbook API
    Replace URL with your actual API provider
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={api_key}&regions=us&markets=h2h,totals,spreads"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching odds: {response.status_code}")
        return []
